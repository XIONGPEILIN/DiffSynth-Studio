"""
Quick LDM-bridge smoke test for Qwen-Image using one dataset_qwen_pe.json entry.

Sample (paths relative to repo root):
{
    "prompt": "Picture 1 is the image to modify. Replace the soccer ball with a brown American football, ensuring the new object has appropriate textures, reflections, and shadows that seamlessly match the existing field and player's lighting and perspective, while maintaining its position relative to the player's foot.",
    "image": "edited/sft/66485.png",
    "edit_image": ["edit_aligned/edit_aligned_10569.png"],
    "ref_gt": "ref_gt_generated/ref_gt_10569.png",
    "back_mask": "ref_gt_generated/mask_combined_10569.png"
}

The script:
- Finds repo root by locating dataset_qwen_pe.json
- Loads the above images
- Loads STE + LoRA weights from train/Qwen-Image-Edit-2509_lora-rank512/step-1000.safetensors
- Runs pipeline with LDM bridge sampling (source -> target)
- Saves ldm_bridge_output.png at repo root
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from safetensors.torch import load_file

REPO_ROOT_SENTINEL = "dataset_qwen_pe.json"


def find_repo_root(start: Path) -> Path:
    """Walk upwards until dataset_qwen_pe.json is found."""
    cur = start
    for _ in range(10):
        if (cur / REPO_ROOT_SENTINEL).exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError(f"Could not locate {REPO_ROOT_SENTINEL} upwards from {start}")


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return Image.open(path).convert("RGB")


def load_mask(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing mask: {path}")
    return Image.open(path).convert("L")


def load_ste_and_lora(pipe, ckpt_path: Path) -> Tuple[int, int]:
    """
    Load STE weights (prefix pipe.ste.) and LoRA tensors into DiT from a safetensors file.
    Returns (num_ste_tensors, num_lora_tensors).
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = load_file(str(ckpt_path), device="cpu")

    ste_prefix = "pipe.ste."
    ste_state = {k[len(ste_prefix):]: v for k, v in state.items() if k.startswith(ste_prefix)}
    if ste_state:
        pipe.ste.load_state_dict(ste_state, strict=False)

    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state)

    return len(ste_state), len(lora_state)


def main():
    repo_root = find_repo_root(Path(__file__).resolve())
    import sys

    sys.path.append(str(repo_root / "DiffSynth-Studio"))

    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

    sample = {
        "prompt": "Picture 1 is the image to modify. Replace the soccer ball with a brown American football, ensuring the new object has appropriate textures, reflections, and shadows that seamlessly match the existing field and player's lighting and perspective, while maintaining its position relative to the player's foot.",
        "image": "edited/sft/66485.png",
        "edit_image": ["edit_aligned/edit_aligned_10569.png"],
        "ref_gt": "ref_gt_generated/ref_gt_10569.png",
        "back_mask": "ref_gt_generated/mask_combined_10569.png",
    }

    image_path = repo_root / sample["image"]
    edit_paths: List[Path] = [repo_root / p for p in sample["edit_image"]]
    back_mask_path = repo_root / sample["back_mask"]

    target_image = load_rgb(image_path)
    edit_images = [load_rgb(p) for p in edit_paths]
    back_mask = load_mask(back_mask_path)

    width, height = target_image.size

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2509",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    # Load STE + LoRA weights from training checkpoint
    ste_lora_ckpt = repo_root / "train/Qwen-Image-Edit-2509_lora-rank512/step-1000.safetensors"
    ste_num, lora_num = load_ste_and_lora(pipe, ste_lora_ckpt)
    print(f"Loaded {ste_num} STE tensors and {lora_num} LoRA tensors from {ste_lora_ckpt}")

    output = pipe(
        prompt=sample["prompt"],
        edit_image=edit_images,
        input_image=target_image,
        back_mask=back_mask,
        height=height,
        width=width,
        num_inference_steps=30,
        ldm_bridge=True,
        cfg_scale=4.0,
        seed=0,
    )

    out_path = repo_root / "ldm_bridge_output.png"
    output.save(out_path)
    print(f"Saved LDM bridge test output to: {out_path}")


if __name__ == "__main__":
    main()
