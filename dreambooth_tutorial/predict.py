import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from omegaconf import OmegaConf

from pkg_resources import resource_filename
config = OmegaConf.load(resource_filename(__name__, "../configs/config.yaml"))
unet = UNet2DConditionModel()
model = torch.load(f"{config.output_dir}/checkpoint-1000/pytorch_model.bin")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
)
pipe.unet.load_state_dict(model)
pipe = pipe.to("cuda")

prompt = "4k, cinematic, a photo of sks bulat"

for i in range(5):
    image = pipe(prompt).images[0]
    image.save(f"test{i}.png")
