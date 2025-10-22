#!/bin/bash

accelerate launch --config_file gp38.yaml  DiffSynth-Studio/examples/qwen_image/model_training/train.py \
  --dataset_base_path "prepared_data_final_testdataset" \
  --dataset_metadata_path "prepared_data_final_testdataset/metadata.json" \
  --data_file_keys "image,edit_image" \
  --height 512 --width 512 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps 1000 \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_v,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 256 \
  --extra_inputs "edit_image" \
  --find_unused_parameters \
  --dataset_num_workers 8 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --gradient_accumulation_steps 1 \
  --output_path "test" \
  --wandb_project "test" \
  --wandb_name "test"