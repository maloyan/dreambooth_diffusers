import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

unet = UNet2DConditionModel()
model = torch.load("/root/dreambooth-diffusers/output/checkpoint-400/pytorch_model.bin")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
)
pipe.unet.load_state_dict(model)
pipe = pipe.to("cuda")

prompt = "4k, cinematic, a photo of sks narek"
image = pipe(prompt).images[0]
image.save("test.png")
