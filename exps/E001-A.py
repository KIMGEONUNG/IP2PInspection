import numpy as np
import os
from os.path import join
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
import sys
import yaml
from omegaconf import OmegaConf
import einops
from einops import rearrange
from pathlib import Path

sys.path.append("./stable_diffusion")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm_re import LatentRestoration
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision.transforms import ToTensor


def PSNR(im1, im2):
    x1 = ToTensor()(im1).permute(2, 1, 0).numpy()
    x2 = ToTensor()(im2).permute(2, 1, 0).numpy()

    return compare_psnr(x1, x2)


def merge_pil(*imgs):
    cat = np.concatenate([np.array(img) for img in imgs], axis=1)
    cat = Image.fromarray(cat)
    return cat


def main():
    """
    Motivation:
    ...
    """
    cfn = Path(os.path.basename(__file__)).stem

    # LOAD CONFIG
    config = OmegaConf.load('configs/generate_highfreq.yaml')
    config.model.params.ckpt_path = "checkpoints/openimage_s/last.ckpt"

    # LOAD MODEL AND DATASET
    model = instantiate_from_config(config.model)
    model.eval().cuda()
    null_token = model.get_learned_conditioning([""])
    instruction = "a high quality, detailed and professional image"
    steps = 50
    seed = 1

    path_dir = join("outputs", cfn)
    os.makedirs(path_dir, exist_ok=True)
    path_metric = join(path_dir, "metric.yaml")
    try:
        os.remove(path_metric)
    except OSError:
        pass
    with torch.no_grad(), model.ema_scope():
        for i, path in enumerate(
                sorted(glob("DATASET/openimage/train200T512x512_de/*.jpg"))[:100]):
            img_gt = Image.open(path.replace('_de',
                                             '')).convert('RGB')

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

            posterior = model.encode_first_stage(input_image)
            cond["c_concat"] = [posterior.mode()]

            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]
            torch.manual_seed(seed)

            z, _ = model.sample_log(
                cond=cond,
                batch_size=1,
                ddim=True,
                eta=0,
                ddim_steps=steps,
            )

            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

            grid = merge_pil(img_lf, edited_image, img_gt)
            grid.save(join(path_dir, f"{i:04}.jpg"),
                      quality=100,
                      sunbsampling=0)

            psnr = PSNR(edited_image, img_gt)

            with open(path_metric, "a") as myfile:
                myfile.write(f"{i:04}: {psnr}\n")


if __name__ == "__main__":
    main()
