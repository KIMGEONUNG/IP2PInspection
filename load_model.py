import math
import random
import statistics
import sys
import time
from argparse import ArgumentParser

import einops
import gradio as gr
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config
from stable_diffusion.ldm.models.diffusion.ddpm_edit import LatentDiffusion


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model."):]]
            if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


if __name__ == "__main__":
    path_config = "configs/generate.yaml"
    path_ckpt = "checkpoints/instruct-pix2pix-00-22000.ckpt"
    # path_config = "configs/lowlight.yaml"
    # path_ckpt = "checkpoints/low-light.ckpt"

    config = OmegaConf.load(path_config)

    x = torch.randn(1, 4, 64, 64).cuda()
    c_concat = [torch.randn(1, 4, 64, 64).cuda()]
    t = torch.LongTensor([100]).cuda()
    c_crossattn = [torch.randn(1, 77, 768).cuda()]
    model: LatentDiffusion = load_model_from_config(config, path_ckpt, None).cuda()

    diffs = []
    for i in range(10):
        start = time.time()
        a = model.model(x, t, c_concat, c_crossattn)
        end = time.time()
        diff = end - start
        diffs.append(diff)
        print("{:.6f}".format(diff))

    print("avg: {:.6f}".format(statistics.mean(diffs)))

    print('finishied')
