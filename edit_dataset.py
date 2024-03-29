from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from os.path import join
import cv2

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToPILImage, ToTensor, Resize, CenterCrop, Compose, GaussianBlur, InterpolationMode, Grayscale
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
            prompt="face, a high quality, detailed and professional image"):
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


class InpaintDataset(Dataset):
    """
    Inline means that the degraded data does not come from I/O but built-in 
    calculation.
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
                 resize_res: int = 512,
                 prompt=""):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.resize_res = resize_res
        self.prompt = prompt
        self.degrader = None

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

    def gen_mask(self,
                 img: Image.Image,
                 length=128,
                 scale=8) -> tuple[Image.Image, torch.Tensor]:
        w, h = img.size
        w_d = w // scale
        h_d = h // scale
        length_d = length // scale
        mask = torch.zeros(1, h_d, w_d)
        off_x = random.randint(0, w_d - length_d - 1)
        off_y = random.randint(0, h_d - length_d - 1)
        mask[:, off_y:off_y + length_d, off_x:off_x + length_d] = 1
        mask_up = Resize((h, w), InterpolationMode.NEAREST)(mask)

        x = ToTensor()(img) * (1 - mask_up)
        img = ToPILImage()(x)
        return img, mask

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        path = self.seeds[i]
        path = join(self.path, path)

        image_1 = Image.open(path).convert('RGB')  # GT
        image_0, mask = self.gen_mask(image_1)

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

        return dict(edited=image_1,
                    edit=dict(c_concat=image_0,
                              c_crossattn=self.prompt,
                              mask=mask))


class GrayscaleDataset(Dataset):
    """
    Inline means that the degraded data does not come from I/O but built-in 
    calculation.
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
                 resize_res: int = 512,
                 flip_prob: float = 0.5,
                 prompt=""):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.resize_res = resize_res
        self.prompt = prompt
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
        self.togray = Grayscale(num_output_channels=3)

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        path = self.seeds[i]
        path = join(self.path, path)

        image_1 = Image.open(path).convert('RGB')  # GT
        image_0 = self.togray(image_1)

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


class NoiseDataset(Dataset):
    """
    Inline means that the degraded data does not come from I/O but built-in 
    calculation.
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
                 resize_res: int = 512,
                 flip_prob: float = 0.5,
                 sigma: float = 0.5,
                 prompt=""):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.resize_res = resize_res
        self.prompt = prompt
        self.sigma = sigma
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

    def add_noise(self, im: Image, sigma: float):
        x = ToTensor()(im)
        n = torch.randn_like(x)
        n.normal_(mean=0, std=sigma)
        x = x + n
        x = x.clamp(0, 1)
        x = ToPILImage()(x)
        return x

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        path = self.seeds[i]
        path = join(self.path, path)

        image_1 = Image.open(path).convert('RGB')  # GT
        image_0 = self.add_noise(image_1, self.sigma)

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


class GaussianBlurDataset(Dataset):
    """
    Inline means that the degraded data does not come from I/O but built-in 
    calculation.
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
                 flip_prob: float = 0.0,
                 resize_res: int = 512,
                 kernel_size: int = 11,
                 sigma: tuple[float, float] = (2.0, 2.0),
                 prompt=""):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.resize_res = resize_res
        self.flip_prob = flip_prob
        self.prompt = prompt
        self.degrader = GaussianBlur(kernel_size, sigma)

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
        image_0 = self.degrader(image_1)

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


class HighFrequencyOpenImageDataset(Dataset):

    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        flip_prob: float = 0.0,
        resize_res: int = 512,
        prompt="a high quality, detailed and professional image",
        filter_d=10,
        filter_sigmaColor=50,
        filter_sigmaSpace=50,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.resize_res = resize_res
        self.flip_prob = flip_prob
        self.prompt = prompt

        self.filter_d = filter_d
        self.filter_sigmaColor = filter_sigmaColor
        self.filter_sigmaSpace = filter_sigmaSpace

        self.sizing = Compose(
            [Resize(self.resize_res),
             CenterCrop(self.resize_res)])

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

    def bilateral_filter(
        self,
        x,
    ):
        x = np.asarray(x)
        x_hat = cv2.bilateralFilter(
            x,
            d=self.filter_d,
            sigmaColor=self.filter_sigmaColor,
            sigmaSpace=self.filter_sigmaSpace,
        )
        x_hat = Image.fromarray(x_hat)
        return x_hat

    def __getitem__(self, i: int) -> dict[str, Any]:
        path = self.seeds[i]
        path = join(self.path, path)

        image_1 = Image.open(path).convert('RGB')  # GT
        image_1 = self.sizing(image_1)

        image_0 = self.bilateral_filter(image_1)

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


class HighFrequencyOpenImagePairDataset(Dataset):

    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.95, 0.05, 0.00),
        resize_res: int = 512,
        prompt="a high quality, detailed and professional image",
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.resize_res = resize_res
        self.prompt = prompt

        self.sizing = Compose(
            [Resize(self.resize_res),
             CenterCrop(self.resize_res)])

        with open(Path(self.path, "seeds_pair.json")) as f:
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
        name = self.seeds[i]
        path_gt = join(self.path, "train200T512x512", name)
        path_re = join(self.path, "train200T512x512_de", name)

        image_0 = Image.open(path_re).convert('RGB')  # I_re
        image_1 = Image.open(path_gt).convert('RGB')  # I_gt

        image_0 = self.sizing(image_0)
        image_1 = self.sizing(image_1)

        image_0 = rearrange(
            2 * torch.tensor(np.array(image_0)).float() / 255 - 1,
            "h w c -> c h w")
        image_1 = rearrange(
            2 * torch.tensor(np.array(image_1)).float() / 255 - 1,
            "h w c -> c h w")

        return dict(edited=image_1,
                    edit=dict(c_concat=image_0, c_crossattn=self.prompt))


if __name__ == "__main__":
    pass
