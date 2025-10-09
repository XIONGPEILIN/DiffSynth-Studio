#!/bin/bash

# Qwen-Image-Edit-2509 Inpaint LoRA Training with Multi-Batch Support
# Batch Size: 2 (适合 24GB GPU)

accelerate launch examples/qwen_image/model_training/train2509.py \
  --dataset_base_path "prepared_data_original" \
  --dataset_metadata_path "prepared_data_original/metadata.json" \
  --data_file_keys "tgt_original_file,tgt_original_mask_file" \
  --height 1024 --width 1024 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors,DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint:model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/Qwen-Image-Edit-2509_inpaint_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 128 \
  --extra_inputs "blockwise_controlnet_image,blockwise_controlnet_inpaint_mask" \
  --find_unused_parameters \
  --dataset_num_workers 8

# 说明:
# - batch_size=2: 每次处理2个样本
# - gradient_accumulation_steps=4: 梯度累积4步
# - 有效批次大小 = 2 * 4 = 8 (等效于batch_size=8但内存占用更少)
