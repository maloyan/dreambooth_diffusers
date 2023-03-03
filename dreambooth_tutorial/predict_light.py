import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from omegaconf import OmegaConf
from pkg_resources import resource_filename

config = OmegaConf.load(resource_filename(__name__, "../configs/config.yaml"))

pipe = StableDiffusionPipeline.from_pretrained(
    config.pretrained_model_name_or_path,
    torch_dtype=torch.float16,
    safety_checker=None
    
)
pipe.unet.load_state_dict(
    torch.load(
        f"{config.output_dir}/{config.instance_prompt}/checkpoint-800/pytorch_model.bin"
    )
)
pipe = pipe.to("cuda")

prompt = "(extremely detailed CG unity 8k wallpaper), full shot body photo of the most beautiful artwork of narek holding a torch, torn jacket, nostalgia professional majestic oil painting by Ed Blinkey, Atey Ghailan, Studio Ghibli, by Jeremy Mann, Greg Manchess, Antonio Moro, trending on ArtStation, trending on CGSociety, Intricate, High Detail, Sharp focus, dramatic, photorealistic painting art by midjourney and greg rutkowski"


image = pipe(prompt).images[0]
image.save(f"test.png")
