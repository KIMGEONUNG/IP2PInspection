from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from os.path import join

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToPILImage, ToTensor
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
import random

from utils import Degrade


class EditDataset(Dataset):

    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        propt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)["edit"]

        image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
        image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1,
                                  ()).item()
        image_0 = image_0.resize((reize_res, reize_res),
                                 Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res),
                                 Image.Resampling.LANCZOS)

        image_0 = rearrange(
            2 * torch.tensor(np.array(image_0)).float() / 255 - 1,
            "h w c -> c h w")
        image_1 = rearrange(
            2 * torch.tensor(np.array(image_1)).float() / 255 - 1,
            "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(
            float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1,
                    edit=dict(c_concat=image_0, c_crossattn=prompt))


class EditDatasetEval(Dataset):

    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        propt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)
            edit = prompt["edit"]
            input_prompt = prompt["input"]
            output_prompt = prompt["output"]

        image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))

        reize_res = torch.randint(self.res, self.res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res),
                                 Image.Resampling.LANCZOS)

        image_0 = rearrange(
            2 * torch.tensor(np.array(image_0)).float() / 255 - 1,
            "h w c -> c h w")

        return dict(image_0=image_0,
                    input_prompt=input_prompt,
                    edit=edit,
                    output_prompt=output_prompt)


class LowlightDataset(Dataset):

    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        path_dark, path_bright = self.seeds[i]
        path_dark, path_bright = join(self.path,
                                      path_dark), join(self.path, path_bright)
        prompt = "make it bright daytime"

        image_0 = Image.open(path_dark)
        image_1 = Image.open(path_bright)

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1,
                                  ()).item()
        image_0 = image_0.resize((reize_res, reize_res),
                                 Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res),
                                 Image.Resampling.LANCZOS)

        image_0 = rearrange(
            2 * torch.tensor(np.array(image_0)).float() / 255 - 1,
            "h w c -> c h w")
        image_1 = rearrange(
            2 * torch.tensor(np.array(image_1)).float() / 255 - 1,
            "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(
            float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1,
                    edit=dict(c_concat=image_0, c_crossattn=prompt))


class DenoiseDataset(Dataset):

    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def add_noise(self, im: Image, sigma: float):
        x = ToTensor()(im)
        n = torch.randn_like(x)
        n.normal_(mean=0, std=sigma)
        x = x + n
        x = x.clamp(0, 1)
        x = ToPILImage()(x)
        return x

    def __getitem__(self, i: int) -> dict[str, Any]:
        path = self.seeds[i]
        path = join(self.path, path)
        prompt = "denoise"

        image_0 = Image.open(path).convert('RGB')
        image_1 = Image.open(path).convert('RGB')

        # Add noise
        sigma = random.random() / 2
        image_0 = self.add_noise(image_0, sigma)

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1,
                                  ()).item()
        image_0 = image_0.resize((reize_res, reize_res),
                                 Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res),
                                 Image.Resampling.LANCZOS)

        image_0 = rearrange(
            2 * torch.tensor(np.array(image_0)).float() / 255 - 1,
            "h w c -> c h w")
        image_1 = rearrange(
            2 * torch.tensor(np.array(image_1)).float() / 255 - 1,
            "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(
            float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1,
                    edit=dict(c_concat=image_0, c_crossattn=prompt))


class DegradationDataset(Dataset):

    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.degrader = Degrade()

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def add_noise(self, im: Image, sigma: float):
        x = ToTensor()(im)
        n = torch.randn_like(x)
        n.normal_(mean=0, std=sigma)
        x = x + n
        x = x.clamp(0, 1)
        x = ToPILImage()(x)
        return x

    def __getitem__(self, i: int) -> dict[str, Any]:
        path = self.seeds[i]
        path = join(self.path, path)

        image_1 = Image.open(path).convert('RGB')
        w, h = image_1.size

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1,
                                  ()).item()

        if w > h:
            image_1 = image_1.resize((int(w / h * reize_res), reize_res),
                                     Image.Resampling.LANCZOS)
        else:
            image_1 = image_1.resize((reize_res, int(h / w * reize_res)),
                                     Image.Resampling.LANCZOS)
        image_0, prompt = self.degrader.random_single_deg(image_1)

        image_0 = rearrange(
            2 * torch.tensor(np.array(image_0)).float() / 255 - 1,
            "h w c -> c h w")
        image_1 = rearrange(
            2 * torch.tensor(np.array(image_1)).float() / 255 - 1,
            "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(
            float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1,
                    edit=dict(c_concat=image_0, c_crossattn=prompt))


class HighFrequencyDataset(Dataset):

    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        flip_prob: float = 0.0,
        resize_res: int = 256,
        prompt = "face, a high quality, detailed and professional image"
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.resize_res = resize_res
        self.flip_prob = flip_prob
        self.prompt = prompt

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        path = self.seeds[i]
        path = join(self.path, path)

        image_1 = Image.open(path).convert('RGB')  # GT
        image_0 = Image.open(path.replace('-512', '-LF')).convert(
            'RGB')  # GT w/o high freq.

        image_0 = image_0.resize((self.resize_res, self.resize_res),
                                 Image.Resampling.LANCZOS)
        image_1 = image_1.resize((self.resize_res, self.resize_res),
                                 Image.Resampling.LANCZOS)

        image_0 = rearrange(
            2 * torch.tensor(np.array(image_0)).float() / 255 - 1,
            "h w c -> c h w")
        image_1 = rearrange(
            2 * torch.tensor(np.array(image_1)).float() / 255 - 1,
            "h w c -> c h w")

        flip = torchvision.transforms.RandomHorizontalFlip(
            float(self.flip_prob))
        image_0, image_1 = flip(torch.cat((image_0, image_1))).chunk(2)

        return dict(edited=image_1,
                    edit=dict(c_concat=image_0, c_crossattn=self.prompt))
