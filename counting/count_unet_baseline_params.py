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

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm_edit import LatentDiffusion
from ldm.modules.attention import SpatialTransformer


def num_param(model):
    return sum(p.numel() for p in model.parameters())


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


def profile_transformer(model):
    print('num original:', num_param(model))
    total = []
    for layer in model.model.diffusion_model.input_blocks:
        for sublayer in layer:
            if isinstance(sublayer, SpatialTransformer):
                print('input_blocks')
                n = num_param(sublayer)
                print('transformer:', n)
                total.append(n)

    for layer in model.model.diffusion_model.middle_block:
        if isinstance(layer, SpatialTransformer):
            print('middle')
            n = num_param(layer)
            print('transformer:', n)
            total.append(n)

    for layer in model.model.diffusion_model.output_blocks:
        for sublayer in layer:
            if isinstance(sublayer, SpatialTransformer):
                print('output_blocks')
                n = num_param(sublayer)
                print('transformer:', n)
                total.append(n)

    print('total:', sum(total))


if __name__ == "__main__":
    path_config = "configs/legacy/generate.yaml"
    path_ckpt = "checkpoints/instruct-pix2pix-00-22000.ckpt"
    # path_config = "configs/lowlight.yaml"
    # path_ckpt = "checkpoints/low-light.ckpt"

    config = OmegaConf.load(path_config)

    x = torch.randn(1, 4, 64, 64).cuda()
    c_concat = [torch.randn(1, 4, 64, 64).cuda()]
    t = torch.LongTensor([100]).cuda()
    c_crossattn = [torch.randn(1, 77, 768).cuda()]
    model: LatentDiffusion = load_model_from_config(config, path_ckpt,
                                                    None).cuda()

    unet = model.model.diffusion_model
    skip_encode_idxs = [2, 5, 8, 9, 10, 11]
    skip_decode_idxs = [0, 1, 2, 4, 7, 10]

    for i in reversed(skip_encode_idxs):
        del unet.input_blocks[i]
    for i in reversed(skip_decode_idxs):
        del unet.output_blocks[i]
    del unet.middle_block

    for layer in unet.input_blocks:
        for sublayer in layer:
            if isinstance(sublayer, SpatialTransformer):
                del sublayer.transformer_blocks[0].attn2
                del sublayer.transformer_blocks[0].norm2

    for layer in unet.output_blocks:
        for sublayer in layer:
            if isinstance(sublayer, SpatialTransformer):
                del sublayer.transformer_blocks[0].attn2
                del sublayer.transformer_blocks[0].norm2

    print('num original:', num_param(unet))
    # total = []
    # for layer in model.model.diffusion_model.input_blocks:
    #     for sublayer in layer:
    #         if isinstance(sublayer, SpatialTransformer):
    #             print('input_blocks')
    #             a = sublayer.transformer_blocks[0].attn2
    #             b = sublayer.transformer_blocks[0].norm2
    #             n = num_param(a) + num_param(b)
    #             print('crossatt:', n)
    #             total.append(n)
    #
    # for sublayer in model.model.diffusion_model.middle_block:
    #     if isinstance(sublayer, SpatialTransformer):
    #         print('middle')
    #         a = sublayer.transformer_blocks[0].attn2
    #         b = sublayer.transformer_blocks[0].norm2
    #         n = num_param(a) + num_param(b)
    #         print('crossatt:', n)
    #         total.append(n)
    #
    # for layer in model.model.diffusion_model.output_blocks:
    #     for sublayer in layer:
    #         if isinstance(sublayer, SpatialTransformer):
    #             print('output_blocks')
    #             a = sublayer.transformer_blocks[0].attn2
    #             b = sublayer.transformer_blocks[0].norm2
    #             n = num_param(a) + num_param(b)
    #             print('crossatt:', n)
    #             total.append(n)
    #
    # print('total:', sum(total))
    # exit()
    #
    # diffs = []
    # for i in range(11):
    #     start = time.time()
    #     a = model.model(x, t, c_concat, c_crossattn)
    #     end = time.time()
    #     diff = end - start
    #     diffs.append(diff)
    #     if 0 == i:
    #         continue
    #     print("{:.6f}".format(diff))
    #
    # print("avg: {:.6f}".format(statistics.mean(diffs[1:])))
    #
    print('finishied')
