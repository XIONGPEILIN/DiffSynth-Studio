import imageio, os, torch, warnings, torchvision, argparse, json, inspect

from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1,
    weight_decay: float = 0,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    wandb_project: str = None,
    wandb_name: str = None,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        wandb_project = args.wandb_project
        wandb_name = args.wandb_name
        save_resume_each_epoch = not getattr(args, "disable_epoch_resume", False)
    else:
        save_resume_each_epoch = True
    
    if learning_rate is None:
        learning_rate = 1.0
    
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=collate_fn=lambda x: x[0], num_workers=num_workers)
    
    # 计算总训练步数
    total_steps = len(dataloader) * num_epochs
    print(f"Total training steps: {total_steps}")
    print("Using Schedule-Free updates; no external LR scheduler will be applied.")
    
    if wandb_project is not None:
        accelerator.init_trackers(project_name=wandb_project, config=vars(args), init_kwargs={"wandb": {"name": wandb_name}})
    
    # -------- 构建更稳健的 PPSF 优化器 --------
    auto_prodigy_steps = min(3000, max(0, int(total_steps * 0.2))) or None  # 前20%，最多3000
    beta1 = getattr(args, "beta1", 0.95)
    beta2 = getattr(args, "beta2", 0.99)
    use_speed = getattr(args, "use_speed", False)
    prodigy_steps = getattr(args, "prodigy_steps", -1)
    prodigy_steps = (None if prodigy_steps in (None, 0, -1) else int(prodigy_steps)) or auto_prodigy_steps
    d0 = getattr(args, "d0", 1e-6)

    def _build_ppsf(params):
        base = dict(lr=learning_rate, weight_decay=weight_decay, use_schedulefree=True)
        extras = dict(betas=(beta1, beta2), use_speed=use_speed, prodigy_steps=prodigy_steps, d0=d0, use_bias_correction=True, safeguard_warmup=True)
        sig = inspect.signature(ProdigyPlusScheduleFree.__init__)
        allowed = set(sig.parameters.keys())
        kw = {k: v for k, v in {**base, **extras}.items() if k in allowed and v is not None}
        return ProdigyPlusScheduleFree(params, **kw)

    optimizer = _build_ppsf(model.trainable_modules())
    print("Optimizer: ProdigyPlusScheduleFree (Schedule‑Free, lr=1.0, use_speed=", use_speed, ", prodigy_steps=", prodigy_steps, ")")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if hasattr(optimizer, "train"):
        optimizer.train()
    

    
    resume_path = os.path.join(args.output_path, "accelerator_state")
    os.makedirs(resume_path, exist_ok=True)


    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}")
        for data in pbar:
            if data is None: continue
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if hasattr(dataset, 'load_from_cache') and dataset.load_from_cache:
                    loss, org_loss, back_loss, sub_loss = model({}, inputs=data)

                else:
                    loss, org_loss, back_loss, sub_loss  = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)

                group0 = optimizer.param_groups[0]
                eff_lr = group0.get("effective_lr", group0.get("lr", learning_rate))
                d_val = group0.get("d", 1.0)
                effective_step = eff_lr * d_val

                if wandb_project is not None:
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "org_loss": org_loss.item() if torch.is_tensor(org_loss) else float(org_loss),
                            "back_loss": back_loss.item() if torch.is_tensor(back_loss) else float(back_loss),
                            "sub_loss": sub_loss.item() if torch.is_tensor(sub_loss) else float(sub_loss),
                            "lr": eff_lr,
                            "effective_step": effective_step,
                        },
                        step=model_logger.num_steps,
                    )
                # Update progress bar with loss and learning rate
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{eff_lr:.2e}"
                })
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        #save resume state
        if save_resume_each_epoch:
            if hasattr(optimizer, "eval"):
                optimizer.eval()
            accelerator.save_state('train/resume')
            if hasattr(optimizer, "train"):
                optimizer.train()
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
