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
    max_steps: int = None,
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
        max_steps = getattr(args, "max_steps", None)
        wandb_project = args.wandb_project
        wandb_name = args.wandb_name
        save_resume_each_epoch = not getattr(args, "disable_epoch_resume", False)
        d0 = getattr(args, "d0", 1e-6)
    else:
        save_resume_each_epoch = True
        d0 = 1e-6
    
    if learning_rate is None:
        learning_rate = 1.0
    
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    # 计算总训练步数
    total_steps = len(dataloader) * num_epochs
    print(f"Total training steps: {total_steps}")
    print("Using Schedule-Free updates; no external LR scheduler will be applied.")

    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
    prodigy_steps = 4000 // gradient_accumulation_steps
    optimizer = ProdigyPlusScheduleFree(model.trainable_modules(), betas=(0.95, 0.99), d0=d0, prodigy_steps=prodigy_steps)
    print(f"Optimizer: ProdigyPlusScheduleFree (Schedule‑Free, lr=1.0, betas=(0.95, 0.99), prodigy_steps={prodigy_steps})")


    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()
    optimizer.train()

    
    resume_path = os.path.join(args.output_path, "accelerator_state")
    os.makedirs(resume_path, exist_ok=True)
    
    if wandb_project is not None:
        accelerator.init_trackers(project_name=wandb_project, config=vars(args), init_kwargs={"wandb": {"name": wandb_name}})
    
    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}")
        stop_training = False
        for data in pbar:
            if data is None: continue
            
            # Check max_steps early stopping
            if max_steps is not None and model_logger.num_steps >= max_steps:
                print(f"\nReached max_steps ({max_steps}). Stopping training.")
                stop_training = True
                break
                
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if hasattr(dataset, 'load_from_cache') and dataset.load_from_cache:
                    loss, org_loss, back_loss, sub_loss = model({}, inputs=data)

                else:
                    loss, org_loss, back_loss, sub_loss  = model(data)
                accelerator.backward(loss)
                optimizer.step()
                if save_steps is not None and (model_logger.num_steps + 1) % save_steps == 0:
                    model.eval()
                    optimizer.eval()
                    model_logger.on_step_end(accelerator, model, save_steps)
                    optimizer.train()
                    model.train()
                else:
                    model_logger.on_step_end(accelerator, model, save_steps=None)

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
            model.eval()
            optimizer.eval()
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            optimizer.train()
            model.train()
        
        # Break out of epoch loop if max_steps reached
        if stop_training:
            break
            
    model.eval()
    optimizer.eval()
    model_logger.on_training_end(accelerator, model, save_steps)
    optimizer.train()
    model.train()


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
