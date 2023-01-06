import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from omegaconf import OmegaConf
from pkg_resources import resource_filename

config = OmegaConf.load(resource_filename(__name__, "../configs/config.yaml"))

pipe = StableDiffusionPipeline.from_pretrained(
    config.pretrained_model_name_or_path,
    torch_dtype=torch.float16,
)
pipe.unet.load_state_dict(
    torch.load(
        f"{config.output_dir}/{config.instance_prompt}/checkpoint-400/pytorch_model.bin"
    )
)
pipe = pipe.to("cuda")

prompt = "4k, cinematic, a photo of sks narek"


image = pipe(prompt).images[0]
image.save(f"test.png")
