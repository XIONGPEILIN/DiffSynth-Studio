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
        lora_rank=32,
        extra_inputs="blockwise_controlnet_image,blockwise_controlnet_inpaint_mask",
        batch_size=4,  # Multi-batch training
    )
    print("--- 0. Using Hardcoded Arguments for Multi-Batch Debugging ---")
    print(f"Batch Size: {args.batch_size}\n")

    # 2. 构建数据集 (与 train2509.py 完全一致)
    print("--- 1. Building Dataset ---")
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
    )
    print("Model loaded successfully.\n")

    # 4. 获取多个数据样本并构建batch
    print(f"--- 3. Fetching {args.batch_size} Samples and Building Batch ---")
    batch_samples = []
    for i in range(min(args.batch_size, len(dataset))):
        sample = dataset[i]
        batch_samples.append(sample)
        print(f"Sample {i+1}:")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  prompt: '{sample.get('prompt', 'N/A')[:50]}...'")
        
        # 打印所有图像类型的字段
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key} type: Tensor, shape: {value.shape}")
            elif isinstance(value, Image.Image):
                print(f"  {key} type: PIL.Image, size: {value.size}")
            elif isinstance(value, list):
                if value and isinstance(value[0], (torch.Tensor, Image.Image)):
                    print(f"  {key} type: List of {type(value[0]).__name__}, length: {len(value)}")
    
    # 构建batch：将多个样本的每个key合并为list
    batch_data = {}
    for key in batch_samples[0].keys():
        batch_data[key] = [sample[key] for sample in batch_samples]
    
    print(f"\nBatch constructed with {len(batch_samples)} samples.\n")

    # 5. 运行预处理
    print("--- 4. Running Batch Preprocessing ---")
    try:
        final_inputs = model.forward_preprocess(batch_data)
        print("Batch preprocessing complete.\n")
        
        # 6. 打印最终的模型输入结构
        print("="*50)
        print("     FINAL BATCH MODEL INPUTS STRUCTURE")
        print("="*50)
        print_structure(final_inputs)
        print("="*50)
        print(f"\nDebug script finished. Batch size: {args.batch_size}")
        print("Check the structure above to verify batch model inputs.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
