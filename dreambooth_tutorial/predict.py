# from dreambooth_tutorial.lora import inject_trainable_lora
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from omegaconf import OmegaConf
from pkg_resources import resource_filename

config = OmegaConf.load(resource_filename(__name__, "../configs/config.yaml"))
model = torch.load(
    f"{config.output_dir}/{config.instance_prompt}/checkpoint-800/pytorch_model.bin"
)

pipe = StableDiffusionPipeline.from_pretrained(
    config.pretrained_model_name_or_path,
    torch_dtype=torch.float16,
    safety_checker=None,
)
# unet_lora_params, _ = inject_trainable_lora(pipe.unet, r=config.lora_rank, loras=None)
pipe.unet.load_state_dict(model)

pipe = pipe.to("cuda")

prompt = "a photo of sks narek"
# prompt = "4k, cinematic, model, greek god, white hair, masculine, mature, handsome, a photo of sks robert"
# prompt = "a photo of sks robert painted portrait of rugged zeus, god of thunder, greek god, white hair, masculine, mature, handsome, upper body, muscular, hairy torso, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by gaston bussiere and alphonse mucha"
# prompt = "figurine, modern Disney style"
# prompt = "a photo of sks angelinaperfectly feminine face!! full body portrait of young fairy earth goddess blessed by nature, floral sunlight crown, light brown hair, symmetrical! intricate, sensual features, dewy skin, reflective skin, highly detailed, divine holy perfection!! digital painting, artstation, concept art, smooth, sharp focus, warm lighting, illustration, art by artgerm and greg rutkowski and alphonse mucha"
# prompt = "a photo of sks angelina gandalf playing with fireworks, lord of the rings, the hobbit, highly detailed, digital art"
# promt = "a photo of sks robert god of the forest, 3 0 years old, rugged, handsome, male, detailed face, clean lines, atmospheric lighting, amazing, full body, thighs, flowers, muscular, intricate, highly detailed, digital painting, deviantart, concept art, sharp focus, illustration, art by greg rutkowski and alphonse mucha"
# prompt = "a photo of sks robert full body portrait character concept art, anime key visual of a confused oldman, studio lighting delicate features finely detailed perfect face directed gaze, gapmoe yandere grimdark, trending on pixiv fanbox, painted by greg rutkowski makoto shinkai takashi takeuchi studio ghibli"
for idx, image in enumerate(pipe(prompt).images):
    image.save(f"result/test{idx}.png")
