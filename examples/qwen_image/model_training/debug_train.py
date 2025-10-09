import torch, os
from PIL import Image

# 确保能从 examples 目录正确导入
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_training.train2509 import QwenImageTrainingModule, ControlNetInput

from diffsynth.trainers.utils import qwen_image_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, ToAbsolutePath, LoadImage


def print_structure(d, indent=0):
    """一个辅助函数，用于漂亮地打印出嵌套数据结构及其形状/类型。"""
    for key, value in d.items():
        print('  ' * indent + f"'{str(key)}'", end=': ')
        if isinstance(value, dict):
            print()
            print_structure(value, indent + 1)
        elif isinstance(value, list):
            if len(value) > 0:
                item = value[0]
                if isinstance(item, torch.Tensor):
                    print(f"List[{len(value)}] of Tensors, first shape: {item.shape}, dtype: {item.dtype}")
                elif isinstance(item, Image.Image):
                     print(f"List[{len(value)}] of PIL.Image, first size: {item.size}")
                elif hasattr(item, '__dict__'): # For objects like ControlNetInput
                    print(f"List[{len(value)}] of {type(item).__name__} objects. First item details:")
                    for k, v in item.__dict__.items():
                        if isinstance(v, Image.Image):
                            print('  ' * (indent + 2) + f"- {k}: PIL.Image with size {v.size}")
                        else:
                            print('  ' * (indent + 2) + f"- {k}: {type(v)}")
                else:
                    print(f"List[{len(value)}] of {type(item)}")
            else:
                print("[] (Empty list)")
        elif isinstance(value, torch.Tensor):
            print(f"Tensor with shape: {value.shape}, dtype: {value.dtype}")
        elif isinstance(value, Image.Image):
            print(f"PIL.Image with size: {value.size}")
        else:
            print(f"{type(value).__name__} ({value})")


if __name__ == "__main__":
    # 0. 检测GPU
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = "cpu"
        print("⚠️  No GPU detected, using CPU")
    
    # 1. 为方便调试，直接在此处硬编码参数
    from types import SimpleNamespace
    args = SimpleNamespace(
        dataset_base_path="prepared_data_original",
        dataset_metadata_path="prepared_data_original/metadata.json",
        data_file_keys="tgt_original_file,tgt_original_mask_file",
        height=1024,
        width=1024,
        max_pixels=1920*1080, # A reasonable default
        model_id_with_origin_paths="Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors,DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint:model.safetensors",
        lora_base_model="dit",
        lora_target_modules="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1",
        lora_rank=128,
        extra_inputs="blockwise_controlnet_image,blockwise_controlnet_inpaint_mask",
        batch_size=4,  # 设置batch大小，1为单样本，>1为多batch
        device=device,  # 使用检测到的device
    )
    print("\n--- 0. Using Hardcoded Arguments for Debugging ---")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch_size}")

    # 2. 构建数据集 (与 train2509.py 完全一致)
    print("\n--- 1. Building Dataset ---")
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1, # For debugging, we don't need repeats
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    print(f"Dataset built successfully. Total samples: {len(dataset)}\n")

    # 3. 加载模型 (与 train2509.py 完全一致)
    print("--- 2. Loading Model ---")
    model = QwenImageTrainingModule(
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        extra_inputs=args.extra_inputs,
        # Provide default values for other potential arguments
        model_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None, lora_checkpoint=None,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        enable_fp8_training=False, task="sft",
        device=args.device,  # 使用检测到的GPU
    )
    print(f"Model loaded successfully on {args.device}.\n")
    
    # 显示GPU内存使用情况
    if args.device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved\n")

    # 4. 获取数据样本并构建batch
    if args.batch_size == 1:
        print("--- 3. Fetching and Preprocessing a Single Sample ---")
        single_data_sample = dataset[0]
        print(f"Sample prompt: '{single_data_sample['prompt'][:80]}...'\n")
        
        # 为了让脚本能处理批次为1的数据，我们将每个值包装在列表中
        batch_sample = {key: [value] for key, value in single_data_sample.items()}
    else:
        print(f"--- 3. Fetching and Building Batch with {args.batch_size} Samples ---")
        batch_samples = []
        actual_batch_size = min(args.batch_size, len(dataset))
        
        for i in range(actual_batch_size):
            sample = dataset[i]
            batch_samples.append(sample)
            prompt_preview = sample['prompt'][:60] + "..." if len(sample['prompt']) > 60 else sample['prompt']
            print(f"  Sample {i+1}: '{prompt_preview}'")
        
        # 构建batch：将多个样本的每个key合并为list
        batch_sample = {}
        for key in batch_samples[0].keys():
            batch_sample[key] = [sample[key] for sample in batch_samples]
        
        print(f"\nBatch constructed with {actual_batch_size} samples.\n")

    # 5. 运行预处理
    print("--- 4. Running Preprocessing ---")
    try:
        final_inputs = model.forward_preprocess(batch_sample)
        print("Preprocessing complete.\n")
        
        # 6. 打印最终的模型输入结构
        print("="*50)
        if args.batch_size == 1:
            print("          FINAL MODEL INPUTS STRUCTURE")
        else:
            print(f"     FINAL BATCH MODEL INPUTS (size={args.batch_size})")
        print("="*50)
        print_structure(final_inputs)
        print("="*50)
        
        # 7. 验证batch维度
        if args.batch_size > 1:
            print("\n--- Batch Dimension Verification ---")
            for key, value in final_inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: batch_dim={value.shape[0]}, expected={min(args.batch_size, len(dataset))}")
        
        # 8. 显示最终GPU内存使用
        if args.device == "cuda":
            print("\n--- Final GPU Memory Usage ---")
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
            print(f"  Current: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            print(f"  Peak: {max_allocated:.2f} GB allocated")
        
        print("\n✅ Debug script finished. Check the structure above to verify model inputs.")
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
