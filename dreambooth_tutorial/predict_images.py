import numpy as np
import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from omegaconf import OmegaConf
from PIL import Image
from pkg_resources import resource_filename
from transformers import (
    AutoTokenizer,
    CLIPFeatureExtractor,
    CLIPModel,
    CLIPProcessor,
    CLIPTextModel,
    CLIPVisionModelWithProjection,
)
from torchvision import transforms  

config = OmegaConf.load(resource_filename(__name__, "../configs/config.yaml"))

# prompt = "4k, cinematic, model, greek god, white hair, masculine, mature, handsome, a photo of sks narek"
num_images_per_prompt = 1
device = "cuda"
batch_size = 1
num_inference_steps = 50
guidance_scale = 3 #7.5
# pretrained_model_path = (
#     f"{config.output_dir}/{config.instance_prompt}/checkpoint-600/pytorch_model.bin"
# )

tokenizer = AutoTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="tokenizer",
    use_fast=False,
    revision=config.revision,
)

text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="text_encoder",
    revision=config.revision,
)

vae = (
    AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=config.revision,
    )
    .to(device)
    .half()
    .eval()
)

unet = (
    UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.revision,
    )
    .to(device)
    .half()
    .eval()
)
#unet.load_state_dict(torch.load(pretrained_model_path))

noise_scheduler = PNDMScheduler.from_pretrained(
    config.pretrained_model_name_or_path, subfolder="scheduler"
)

# instance_prompt_ids = tokenizer(
#     prompt,
#     truncation=True,
#     padding="max_length",
#     max_length=tokenizer.model_max_length,
#     return_tensors="pt",
# ).input_ids

# uncond_tokens_ids = tokenizer(
#     "",
#     truncation=True,
#     padding="max_length",
#     max_length=tokenizer.model_max_length,
#     return_tensors="pt",
# ).input_ids

# encoder_hidden_states = (
#     torch.cat(
#         [
#             text_encoder(uncond_tokens_ids)[0],
#             text_encoder(instance_prompt_ids)[0],
#         ]
#     )
#     .to(device)
#     .half()
# )
vision_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")

images = [
    #Image.open("/root/dreambooth-diffusers/instance_data/angelina/Screenshot 2022-12-30 at 16.16.02.png"),
    torch.zeros((224, 224, 3)),
    Image.open("/root/dreambooth-diffusers/instance_data/narek/narek_(3).jpg")
]

inputs = processor(
    images=images, return_tensors="pt"
)
image_embedding = vision_encoder.get_image_features(**inputs).unsqueeze(1).cuda().half()

# encoder_hidden_states = (
#     torch.cat(
#         [
#             torch.zeros_like(image_embedding),
#             image_embedding,
#         ]
#     )
#     .to(device)
#     .half()
# )
encoder_hidden_states = image_embedding
noise_scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = noise_scheduler.timesteps

num_channels_latents = unet.in_channels
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

shape = (1, num_channels_latents, unet.config.sample_size, unet.config.sample_size)
latents = (
    torch.randn(shape, device=device, dtype=encoder_hidden_states.dtype)
    .to(device)
    .half()
)
latents = latents * noise_scheduler.init_noise_sigma

# 7. Denoising loop
num_warmup_steps = len(timesteps) - num_inference_steps * noise_scheduler.order
with torch.no_grad():
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = unet(
            latent_model_input, t, encoder_hidden_states=encoder_hidden_states
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        

latents = 1 / 0.18215 * latents
image = vae.decode(latents).sample
image = (image / 2 + 0.5).clamp(0, 1)
# we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
Image.fromarray((image[0] * 255).astype(np.uint8)).save("test.png")
