import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, qwen_image_parser, launch_training_task, launch_data_process_task
from diffsynth.trainers.unified_dataset import UnifiedDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        enable_fp8_training=False,
        task="sft",
        device="cpu",  # 初始化时用cpu，accelerate会在训练时自动移到GPU
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=enable_fp8_training)
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=enable_fp8_training,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.task = task

    
    def forward_preprocess(self, data):
        # Map dataset keys to the keys expected by the pipeline
        key_mapping = {
            "image": "tgt_original_file",
            "blockwise_controlnet_image": "tgt_original_file",
            "blockwise_controlnet_inpaint_mask": "tgt_original_mask_file",
        }
        mapped_data = data.copy()
        for target_key, source_key in key_mapping.items():
            if source_key in data:
                mapped_data[target_key] = data[source_key]
        data = mapped_data

        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": [""] * len(data["prompt"])}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"][0].size[1],
            "width": data["image"][0].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
        }
        
        # Extra inputs
        controlnet_input, blockwise_controlnet_input = {}, {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("blockwise_controlnet_"):
                blockwise_controlnet_input[extra_input.replace("blockwise_controlnet_", "")] = data[extra_input]
            elif extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # 为批处理创建多个 ControlNetInput（每个样本一个）
        batch_size = len(data["prompt"])
        if len(controlnet_input) > 0:
            # 创建 batch_size 个 ControlNetInput
            controlnet_inputs_list = []
            for i in range(batch_size):
                single_input = {k: (v[i] if isinstance(v, list) else v) for k, v in controlnet_input.items()}
                controlnet_inputs_list.append(ControlNetInput(**single_input))
            inputs_shared["controlnet_inputs"] = controlnet_inputs_list
        
        if len(blockwise_controlnet_input) > 0:
            # 创建 batch_size 个 ControlNetInput
            blockwise_controlnet_inputs_list = []
            for i in range(batch_size):
                single_input = {k: (v[i] if isinstance(v, list) else v) for k, v in blockwise_controlnet_input.items()}
                blockwise_controlnet_inputs_list.append(ControlNetInput(**single_input))
            inputs_shared["blockwise_controlnet_inputs"] = blockwise_controlnet_inputs_list
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None, return_inputs=False):
        # Inputs
        if inputs is None:
            inputs = self.forward_preprocess(data)
        else:
            inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        if return_inputs: return inputs
        
        # Loss
        if self.task == "sft":
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            loss = self.pipe.training_loss(**models, **inputs)
        elif self.task == "data_process":
            loss = inputs
        elif self.task == "direct_distill":
            loss = self.pipe.direct_distill_loss(**inputs)
        else:
            raise NotImplementedError(f"Unsupported task: {self.task}.")
        return loss



if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        enable_fp8_training=args.enable_fp8_training,
        task=args.task,
    )
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launcher_map = {
        "sft": launch_training_task,
        "data_process": launch_data_process_task,
        "direct_distill": launch_training_task,
    }
    launcher_map[args.task](dataset, model, model_logger, args=args)








# json：
# [
#   {
#     "id": "0000749a25d7882bad58e6a34200804db1da5a5ff704abf96b96f17688c03f4d",
#     "prompt": "At the foot of a rainforest trail under heavy overcast skies, it rests on dewy leaves, with towering green foliage all around.",
#     "ref_file": "0000749a25d7882bad58e6a34200804db1da5a5ff704abf96b96f17688c03f4d.png",
#     "ref_mask_file": null,
#     "ref_mask_type": null,
#     "tgt_original_file": "0000749a25d7882bad58e6a34200804db1da5a5ff704abf96b96f17688c03f4d.png",
#     "tgt_original_mask_file": "0000749a25d7882bad58e6a34200804db1da5a5ff704abf96b96f17688c03f4d_mask.png"
#   },
#   {
#     "id": "0000e237dc054cf71bcdfe31bde068483aa3cb074e178ef5f4a685763d48e2e7",
#     "prompt": "During a rainy afternoon indoors, it rests coiled by the entrance, with droplets visible on a nearby window, overlooking a street where people hurriedly walk with umbrellas.",
#     "ref_file": "0000e237dc054cf71bcdfe31bde068483aa3cb074e178ef5f4a685763d48e2e7.png",
#     "ref_mask_file": null,
#     "ref_mask_type": null,
#     "tgt_original_file": "0000e237dc054cf71bcdfe31bde068483aa3cb074e178ef5f4a685763d48e2e7.png",
#     "tgt_original_mask_file": "0000e237dc054cf71bcdfe31bde068483aa3cb074e178ef5f4a685763d48e2e7_mask.png"
#   },
#   {
#     "id": "00011c7fd1b57974053345938d1b9867b543d04eacb3fb3c872df02c622bda28",
#     "prompt": "Inside a kitchen, placed next to a mixing bowl, it is shot from above under warm artificial lighting, with baking supplies scattered around on the countertop.",
#     "ref_file": "00011c7fd1b57974053345938d1b9867b543d04eacb3fb3c872df02c622bda28.png",
#     "ref_mask_file": null,
#     "ref_mask_type": null,
#     "tgt_original_file": "00011c7fd1b57974053345938d1b9867b543d04eacb3fb3c872df02c622bda28.png",
#     "tgt_original_mask_file": "00011c7fd1b57974053345938d1b9867b543d04eacb3fb3c872df02c622bda28_mask.png"
#   },
#   {
#     "id": "00016896889fc32eacb48fe488834f1d51dcc23bda000a36fbe1fbbc63ecd114",
#     "prompt": "It waits on a suburban sidewalk, seen in profile with cloudy skies looming large, leaves swirling gently in the autumn breeze around it.",
#     "ref_file": "00016896889fc32eacb48fe488834f1d51dcc23bda000a36fbe1fbbc63ecd114.png",
#     "ref_mask_file": null,
#     "ref_mask_type": null,
#     "tgt_original_file": "00016896889fc32eacb48fe488834f1d51dcc23bda000a36fbe1fbbc63ecd114.png",
#     "tgt_original_mask_file": "00016896889fc32eacb48fe488834f1d51dcc23bda000a36fbe1fbbc63ecd114_mask.png"
#   },