from .base_pipeline import BasePipeline
import torch


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    # 1. Sample Timestep
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    # 2. Add Noise (Main Image Only)
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    # [ABLATION] Removed sub_input_latents handling

    # 3. Model Forward
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    
    # pipe.model_fn will handle zero_cond_t internally if configured
    # We expect model_fn to return (noise_pred, None) after our pipeline modifications
    noise_pred, _ = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    # 4. Compute Loss
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    
    org_loss = loss.clone()
    loss = loss * pipe.scheduler.training_weight(timestep)
    
    # Return dummy zeros for back_loss and sub_loss to match runner expectations
    return loss, org_loss, loss, torch.tensor(0.0, device=loss.device)


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss
