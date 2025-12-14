export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen-image}"
export WANDB_NAME="${WANDB_NAME:-Qwen-Image-Edit-2509-full}"

RESUME_EPOCH="${RESUME_EPOCH:-1}" # 1=保存每个epoch的resume，0=关闭
RESUME_FLAG=""
if [ "$RESUME_EPOCH" = "0" ]; then
  RESUME_FLAG="--disable_epoch_resume"
fi

accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_qwen_imgae_edit_multi.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Edit-2509_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  --find_unused_parameters \
  ${RESUME_FLAG}
