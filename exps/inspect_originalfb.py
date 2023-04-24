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
    In previous experiment, inspect_shortcutgt.py, we found a bug and fix it.
    However, the bug imply that our Forward-Backward trick expriment was conducted
    including the bug. In this experiment, we try Forward-Backward trick 
    without the bug.
    """

    # LOAD CONFIG
    config = OmegaConf.load('configs/generate_highfreq.yaml')
    config.model.params.ckpt_path = "checkpoints/openimage_s/last.ckpt"

    # LOAD MODEL AND DATASET
    model: LatentRestoration = instantiate_from_config(config.model)
    # model.first_stage_model.init_from_ckpt("checkpoints/fuserfeat.ckpt")
    model.eval().cuda()
    null_token = model.get_learned_conditioning([""])
    instruction = "a high quality, detailed and professional image"
    steps = 50
    seed = 1

    path_dir = "outputs/output_originalfb"
    os.makedirs(path_dir, exist_ok=True)
    path_metric = join(path_dir, "metric.yaml")
    try:
        os.remove(path_metric)
    except OSError:
        pass
    with torch.no_grad(), model.ema_scope():
        for i, path in enumerate(
                sorted(glob("DATASET/transform_lf_test/*.jpg"))[:100]):

            if i not in [6, 10, 39]:
                continue

            img_gt = Image.open(path.replace('transform_lf',
                                             'transform')).convert('RGB')
            input_gt = Image.open(path.replace('transform_lf',
                                               'transform')).convert('RGB')

            input_image = Image.open(path).convert('RGB')
            img_lf = Image.open(path).convert('RGB')

            cond = {}
            cond["c_crossattn"] = [
                model.get_learned_conditioning([instruction])
            ]

            input_image = 2 * torch.tensor(
                np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image,
                                    "h w c -> 1 c h w").to(model.device)

            input_gt = 2 * torch.tensor(np.array(input_gt)).float() / 255 - 1
            input_gt = rearrange(input_gt, "h w c -> 1 c h w").to(model.device)

            posterior_gt = model.encode_first_stage(input_gt)
            posterior = model.encode_first_stage(input_image)

            cond["c_concat"] = [posterior.mode()]

            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            t_start = 500
            t = torch.IntTensor([t_start]).to(torch.int64).cuda()

            # FROM GT
            target_p = posterior_gt
            torch.manual_seed(seed)
            x_start = target_p.mode()
            x_start = x_start * model.scale_factor
            x_T = model.q_sample(x_start=x_start, t=t)  # TARGET POSITION

            z, _ = model.sample_log(
                cond=cond,
                batch_size=1,
                ddim=False,
                eta=0,
                start_T=t_start,
                ddim_steps=steps,
                x_T=x_T,
            )
            x = model.first_stage_model.decode(z / model.scale_factor)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_from_gt = Image.fromarray(x.type(torch.uint8).cpu().numpy())

            # FROM LF
            target_p = posterior
            torch.manual_seed(seed)
            x_start = target_p.mode()
            x_start = x_start * model.scale_factor
            x_T = model.q_sample(x_start=x_start, t=t)  # TARGET POSITION

            z, _ = model.sample_log(
                cond=cond,
                batch_size=1,
                ddim=False,
                eta=0,
                start_T=t_start,
                ddim_steps=steps,
                x_T=x_T,
            )

            x = model.first_stage_model.decode(z / model.scale_factor)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_from_lf = Image.fromarray(x.type(torch.uint8).cpu().numpy())

            # SAVE IMAGE
            grid = merge_pil(edited_from_gt, edited_from_lf, img_gt, img_lf)
            grid.save(join(path_dir, f"{i:04}.jpg"),
                      quality=100,
                      sunbsampling=0)

            psnr_gt = PSNR(edited_from_gt, img_gt)
            psnr_lf  = PSNR(edited_from_lf, img_gt)

            with open(path_metric, "a") as myfile:
                myfile.write(f"start@{t_start} {i:04}_gt: {psnr_gt}\n")
                myfile.write(f"start@{t_start} {i:04}_lf: {psnr_lf}\n")


if __name__ == "__main__":
    main()
