import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
import sys
from omegaconf import OmegaConf
import einops
from einops import rearrange
import k_diffusion as K

sys.path.append("./stable_diffusion")
from ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [
                torch.cat([
                    cond["c_crossattn"][0], uncond["c_crossattn"][0],
                    uncond["c_crossattn"][0]
                ])
            ],
            "c_concat": [
                torch.cat([
                    cond["c_concat"][0], cond["c_concat"][0],
                    uncond["c_concat"][0]
                ])
            ],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(
            cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (
            out_cond - out_img_cond) + image_cfg_scale * (out_img_cond -
                                                          out_uncond)


def merge_pil(*imgs):
    cat = np.concatenate([np.array(img) for img in imgs], axis=1)
    cat = Image.fromarray(cat)
    return cat

def main():
    # LOAD CONFIG
    config = OmegaConf.load('configs/generate_baseline.yaml')
    config.model.params.ckpt_path = "checkpoints/openimage_s/last.ckpt"

    # LOAD MODEL AND DATASET
    model = instantiate_from_config(config.model)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    instruction = "a high quality, detailed and professional image"
    steps = 50
    text_cfg_scale = 7.5
    image_cfg_scale = 1.5
    seed = 1

    with torch.no_grad(), model.ema_scope():
        for i, path in enumerate(sorted(glob("DATASET/transform_lf/*.jpg"))):
            img_gt = Image.open(path.replace('transform_lf', 'transform')).convert('RGB')
            input_gt = Image.open(path.replace('transform_lf', 'transform')).convert('RGB')

            img_lf = Image.open(path).convert('RGB')
            input_image = Image.open(path).convert('RGB')

            cond = {}
            cond["c_crossattn"] = [
                model.get_learned_conditioning([instruction])
            ]

            input_image = 2 * torch.tensor(
                np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image,
                                    "h w c -> 1 c h w").to(model.device)

            input_gt = 2 * torch.tensor(
                np.array(input_gt)).float() / 255 - 1
            input_gt = rearrange(input_gt,
                                    "h w c -> 1 c h w").to(model.device)

            posterior, intermediate = model.encode_first_stage(input_image)
            posterior_gt, _ = model.encode_first_stage(input_gt)
            cond["c_concat"] = [posterior.mode()]

            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": text_cfg_scale,
                "image_cfg_scale": image_cfg_scale,
            }
            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg,
                                                  z,
                                                  sigmas,
                                                  extra_args=extra_args)
            x = model.first_stage_model.decode(z, intermediate)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

            x = model.first_stage_model.decode(posterior_gt.mode(), intermediate)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            original_result = Image.fromarray(x.type(torch.uint8).cpu().numpy())

            grid = merge_pil(img_lf, edited_image, img_gt, original_result)
            grid.save(f"output_baseline/{i}.jpg", quality=100, sunbsampling=0)


if __name__ == "__main__":
    main()
