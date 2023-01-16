import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from omegaconf import OmegaConf
from pkg_resources import resource_filename
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms
device = "cuda:0"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
)
sd_pipe = sd_pipe.to(device)

im = Image.open("/root/dreambooth-diffusers/instance_data/narek/narek_(5).jpg")
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])
inp = tform(im).to(device).unsqueeze(0)
out = sd_pipe(inp, guidance_scale=7.5)
out["images"][0].save("test.png")


# config = OmegaConf.load(resource_filename(__name__, "../configs/config.yaml"))

# pipe = StableDiffusionPipeline.from_pretrained(
#     config.pretrained_model_name_or_path,
#     torch_dtype=torch.float16,
# )
# pipe.unet.load_state_dict(
#     torch.load(
#         f"{config.output_dir}/{config.instance_prompt}/checkpoint-400/pytorch_model.bin"
#     )
# )
# pipe = pipe.to("cuda")

# prompt = "4k, cinematic, a photo of sks narek"


# image = pipe(prompt).images[0]
# image.save(f"test.png")
