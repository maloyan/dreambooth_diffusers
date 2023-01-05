import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from omegaconf import OmegaConf
from pkg_resources import resource_filename


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, scaling_rank, init_scale):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        if self.rank > 0:
            self.lora_a = nn.Parameter(
                torch.randn(rank, linear_layer.in_features) * init_scale
            )
            if init_scale < 0:
                self.lora_b = nn.Parameter(
                    torch.randn(linear_layer.out_features, rank) * init_scale
                )
            else:
                self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale
            )
            if init_scale < 0:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                    + torch.randn(linear_layer.out_features, self.scaling_rank)
                    * init_scale
                )
            else:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                )

    def forward(self, input):
        if self.scaling_rank == 1 and self.rank == 0:
            # parsimonious implementation for ia3 and lora scaling
            if self.multi_lora_a.requires_grad:
                hidden = F.linear(
                    (input * self.multi_lora_a.flatten()), self.weight, self.bias
                )
            else:
                hidden = F.linear(input, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                hidden = hidden * self.multi_lora_b.flatten()
            return hidden
        else:
            # general implementation for lora (adding and scaling)
            weight = self.weight
            if self.scaling_rank:
                weight = (
                    weight
                    * torch.matmul(self.multi_lora_b, self.multi_lora_a)
                    / self.scaling_rank
                )
            if self.rank:
                weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
            return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return (
            "in_features={}, out_features={}, bias={}, rank={}, scaling_rank={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.rank,
                self.scaling_rank,
            )
        )


def modify_with_lora(model, config):
    for m_name, module in dict(model.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(
                            layer,
                            config.lora_rank,
                            config.lora_scaling_rank,
                            config.lora_init_scale,
                        ),
                    )
    return model


config = OmegaConf.load(resource_filename(__name__, "../configs/config.yaml"))

model = torch.load(
    f"{config.output_dir}/{config.instance_prompt}/checkpoint-400/pytorch_model.bin"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
)

pipe.unet = modify_with_lora(pipe.unet, config)
pipe.unet.load_state_dict(model)
pipe.unet = pipe.unet.half()

pipe = pipe.to("cuda")

prompt = "a photo of sks narek"
#prompt = "4k, cinematic, model, greek god, white hair, masculine, mature, handsome, a photo of sks robert"
# prompt = "a photo of sks robert painted portrait of rugged zeus, god of thunder, greek god, white hair, masculine, mature, handsome, upper body, muscular, hairy torso, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by gaston bussiere and alphonse mucha"
# prompt = "figurine, modern Disney style"
# prompt = "a photo of sks angelinaperfectly feminine face!! full body portrait of young fairy earth goddess blessed by nature, floral sunlight crown, light brown hair, symmetrical! intricate, sensual features, dewy skin, reflective skin, highly detailed, divine holy perfection!! digital painting, artstation, concept art, smooth, sharp focus, warm lighting, illustration, art by artgerm and greg rutkowski and alphonse mucha"
# prompt = "a photo of sks angelina gandalf playing with fireworks, lord of the rings, the hobbit, highly detailed, digital art"
# promt = "a photo of sks robert god of the forest, 3 0 years old, rugged, handsome, male, detailed face, clean lines, atmospheric lighting, amazing, full body, thighs, flowers, muscular, intricate, highly detailed, digital painting, deviantart, concept art, sharp focus, illustration, art by greg rutkowski and alphonse mucha"
#prompt = "a photo of sks narek full body portrait character concept art, anime key visual of a confused oldman, studio lighting delicate features finely detailed perfect face directed gaze, gapmoe yandere grimdark, trending on pixiv fanbox, painted by greg rutkowski makoto shinkai takashi takeuchi studio ghibli"
for idx, image in enumerate(pipe(prompt).images):
    image.save(f"result/test{idx}.png")
