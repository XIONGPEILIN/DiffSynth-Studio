import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, ControlNetInput
from diffsynth.diffusion import *
from diffsynth.core.data.operators import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        zero_cond_t=False,
        cfg_drop_prob=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.zero_cond_t = zero_cond_t
        self.cfg_drop_prob = cfg_drop_prob
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
            "zero_cond_t": self.zero_cond_t,
            "tiled": False,
            "tile_size": 128,
            "tile_stride": 64,
        }
        # Assume you are using this pipeline for inference,
        # please fill in the input parameters.
        if "image" in data:
            if isinstance(data["image"], list):
                inputs_shared.update({
                    "input_image": data["image"],
                    "height": data["image"][0].size[1],
                    "width": data["image"][0].size[0],
                })
            else:
                inputs_shared.update({
                    "input_image": data["image"],
                    "height": data["image"].size[1],
                    "width": data["image"].size[0],
                })
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)

        # Mapping for ControlNet training using custom dataset keys
        if "edit_image" in inputs_shared:
            img = inputs_shared["edit_image"]
            if isinstance(img, list): img = img[0]
            mask = inputs_shared.get("back_mask")
            if isinstance(mask, list): mask = mask[0]
            if mask is None and "mask" in data: mask = data["mask"]
            inputs_shared["blockwise_controlnet_inputs"] = [ControlNetInput(image=img, inpaint_mask=mask)]

        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        
        # Unpack inputs
        inputs_shared, inputs_posi, inputs_nega = inputs
        
        # CRITICAL FIX: Copy inputs_shared to prevent side-effects impacting Checkpoint Recompute
        inputs_shared = inputs_shared.copy()
        
        # Disable Manual GC Offload (Let DeepSpeed handle it via yaml config)
        # inputs_shared["use_gradient_checkpointing_offload"] = True

        # FORCE DISABLE GC (Switching to ZeRO-2 Offload strategy)
        inputs_shared["use_gradient_checkpointing"] = False
        
        # Inject ControlNet inputs (Run on COPY)
        # Note: logic relies on "edit_latents" being present (which it is in cache)
        if "edit_latents" in inputs_shared:
            if os.getenv("RANK", "0") == "0":
                pass 
            
            edit_l = inputs_shared["edit_latents"]
            if isinstance(edit_l, list): edit_l = edit_l[0]
            
            # Check for mask
            mask_l = inputs_shared.get("processed_inpaint_mask")
            if mask_l is not None:
                if isinstance(mask_l, list): mask_l = mask_l[0]
                if mask_l.shape[-2:] != edit_l.shape[-2:]:
                    mask_l = torch.nn.functional.interpolate(mask_l, size=edit_l.shape[-2:])
                
                mask_final = 1 - mask_l
                combined = torch.cat([edit_l, mask_final], dim=1)
                
                # RE-CREATE input list to ensure fresh objects
                inputs_shared["blockwise_controlnet_inputs"] = [ControlNetInput(image=None)]
                
                # MANUAL PREPROCESS
                cond_list = [combined]
                inp_list = inputs_shared["blockwise_controlnet_inputs"]
                
                try:
                    if hasattr(self.pipe, "blockwise_controlnet") and self.pipe.blockwise_controlnet is not None:
                        processed = self.pipe.blockwise_controlnet.preprocess(inp_list, cond_list)
                        inputs_shared["blockwise_controlnet_conditioning"] = processed
                    else:
                        inputs_shared["blockwise_controlnet_conditioning"] = cond_list
                except Exception as e:
                    inputs_shared["blockwise_controlnet_conditioning"] = cond_list

            else:
                inputs_shared["blockwise_controlnet_conditioning"] = [edit_l]

        # Pack the modified copy back into inputs tuple
        inputs = (inputs_shared, inputs_posi, inputs_nega)

        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        
        inputs_shared, inputs_posi, inputs_nega = inputs
        if self.training and self.cfg_drop_prob > 0:
            if torch.rand(1).item() < self.cfg_drop_prob:
                inputs_posi = inputs_posi.copy()
                inputs_posi["prompt_emb"] = inputs_nega["prompt_emb"]
                inputs_posi["prompt_emb_mask"] = inputs_nega["prompt_emb_mask"]
                inputs = (inputs_shared, inputs_posi, inputs_nega)

        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name.")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name.")
    parser.add_argument("--disable_epoch_resume", action="store_true", help="If set, skip saving accelerator state each epoch (no resume checkpoints).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to resume checkpoint. If provided, training will resume from this checkpoint.")
    parser.add_argument("--cfg_drop_prob", type=float, default=0.0, help="Probability of dropping the text condition (CFG training).")
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="A special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.")
    return parser


if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.wandb_project is not None else None,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
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
        ),
        special_operator_map={
            "ref_gt": ToAbsolutePath(args.dataset_base_path) >> LoadImage(convert_RGB=True),
        },
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
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
        cfg_drop_prob=args.cfg_drop_prob,
        zero_cond_t=args.zero_cond_t,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
