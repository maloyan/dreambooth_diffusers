import os

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from dreambooth_tutorial.dataset import DreamBoothDataset
from omegaconf import OmegaConf
from pkg_resources import resource_filename
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel

config = OmegaConf.load(resource_filename(__name__, "../configs/config.yaml"))

accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision=config.mixed_precision,
    log_with="wandb",
)

if accelerator.is_main_process:
    wandb.init(
        config=config,
        project="diffusers",
        name=f"{config['pretrained_model_name_or_path']}",
    )

tokenizer = AutoTokenizer.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_fast=False,
    revision=config.revision,
)

text_encoder = CLIPTextModel.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="text_encoder",
    revision=config.revision,
)
vae = AutoencoderKL.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="vae",
    revision=config.revision,
)
unet = UNet2DConditionModel.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="unet",
    revision=config.revision,
)

text_encoder.requires_grad_(False)
vae.requires_grad_(False)

optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=config.learning_rate,
)

noise_scheduler = PNDMScheduler.from_pretrained(
    config.pretrained_model_name_or_path, subfolder="scheduler"
)

train_dataset = DreamBoothDataset(
    instance_data_root=config.instance_data_dir,
    instance_prompt=f"a photo of sks {config.instance_prompt}",
    tokenizer=tokenizer,
    size=config.resolution,
    center_crop=config.center_crop,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.train_batch_size,
    shuffle=False,
    num_workers=4,
)

lr_scheduler = get_scheduler(
    config.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
    num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
)

(
    unet,
    text_encoder,
    vae,
    optimizer,
    train_dataloader,
    lr_scheduler,
) = accelerator.prepare(
    unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler
)
accelerator.register_for_checkpointing(lr_scheduler)

progress_bar = tqdm(
    range(config.max_train_steps),
    disable=not accelerator.is_local_main_process,
)
progress_bar.set_description("Steps")
global_step = 0

unet.train()
for _ in progress_bar:
    for batch in train_dataloader:
        with accelerator.accumulate(unet):
            # Convert images to latent space
            latents = vae.encode(batch["instance_images"]).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["instance_prompt_ids"])[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            if config.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = (
                    F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                # Compute prior loss
                prior_loss = F.mse_loss(
                    model_pred_prior.float(), target_prior.float(), reduction="mean"
                )

                # Add the prior loss to the instance loss.
                loss = loss + config.prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = ()
                accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if global_step % 100 == 0 and accelerator.is_main_process:
                save_path = os.path.join(
                    config.output_dir,
                    config.instance_prompt,
                    f"checkpoint-{global_step}",
                )
                accelerator.save_state(save_path)

        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "global_step": global_step,
        }
        progress_bar.set_postfix(**logs)
        if accelerator.is_main_process:
            wandb.log(logs)

    if global_step >= config.max_train_steps:
        break

    accelerator.wait_for_everyone()

accelerator.end_training()
