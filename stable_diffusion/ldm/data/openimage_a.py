from __future__ import annotations

from os.path import join

import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor, Resize, CenterCrop, Compose, RandomCrop
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


class OpenImagePairDataset(Dataset):

    def __init__(
        self,
        path_gt: str,
        path_in: str,
        path_idx: str,
        resize_res: int = 512,
        size_crop: int = 512,
        prompt="a high quality, detailed and professional image",
    ):
        self.prompt = prompt

        self.path_gt = path_gt
        self.path_in = path_in
        self.path_idx = path_idx

        self.resize_res = resize_res
        self.size_crop = size_crop

        self.sizing = Compose(
            [Resize(self.resize_res),
             RandomCrop(self.size_crop)])

        with open(self.path_idx) as f:
            self.seeds = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int):
        name = self.seeds[i]
        path_gt = join(self.path_gt, name)
        path_in = join(self.path_in, name)

        img_in = Image.open(path_in).convert('RGB')  # I_re
        img_gt = Image.open(path_gt).convert('RGB')  # I_gt

        # MERGE & CROP & AND & SPLIT
        merge = torch.cat([ToTensor()(img_in), ToTensor()(img_gt)])
        img_in, img_gt = self.sizing(merge).chunk(2)
        img_in, img_gt = ToPILImage()(img_in), ToPILImage()(img_gt)

        img_in = rearrange(
            2 * torch.tensor(np.array(img_in)).float() / 255 - 1,
            "h w c -> c h w")
        img_gt = rearrange(
            2 * torch.tensor(np.array(img_gt)).float() / 255 - 1,
            "h w c -> c h w")

        return dict(edited=img_gt,
                    edit=dict(c_concat=img_in, c_crossattn=self.prompt))
