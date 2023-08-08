import argparse, os, sys, datetime, glob
from os.path import join
from pathlib import Path
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import json
import pickle

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from functools import partial
from PIL import Image

import torch.distributed as dist
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.plugins import DDPPlugin

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id *
                                               split_size:(worker_id + 1) *
                                               split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 train=None,
                 validation=None,
                 test=None,
                 predict=None,
                 wrap=False,
                 num_workers=None,
                 shuffle_test_loader=False,
                 use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader,
                                          shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader,
                                           shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'],
                                         Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn,
                          persistent_workers=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'],
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle,
                          persistent_workers=True)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'],
                                         Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle,
                          persistent_workers=True)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'],
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          persistent_workers=True)


class SetupCallback(Callback):

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config,
                 lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    # def on_keyboard_interrupt(self, trainer, pl_module):
    #     if trainer.global_rank == 0:
    #         print("Summoning checkpoint.")
    #         ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
    #         trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            # os.makedirs(self.logdir, exist_ok=True)
            # os.makedirs(self.ckptdir, exist_ok=True)
            # os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config[
                        'callbacks']:
                    os.makedirs(os.path.join(self.ckptdir,
                                             'trainstep_checkpoints'),
                                exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir,
                             "{}-lightning.yaml".format(self.now)))


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    origin_size = None
    if not isinstance(data, torch.Tensor):
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")
    else:
        origin_size = data.size()
        tensor = data.reshape(-1)

    tensor_type = tensor.dtype

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.FloatTensor(size=(max_size, )).cuda().to(tensor_type))
    if local_size != max_size:
        padding = torch.FloatTensor(size=(max_size -
                                          local_size, )).cuda().to(tensor_type)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        if origin_size is None:
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        else:
            buffer = tensor[:size]
            data_list.append(buffer)

    if origin_size is not None:
        new_shape = [-1] + list(origin_size[1:])
        resized_list = []
        for data in data_list:
            # suppose the difference of tensor size exist in first dimension
            data = data.reshape(new_shape)
            resized_list.append(data)

        return resized_list
    else:
        return data_list


class ImageLogger(Callback):

    def __init__(self,
                 batch_frequency,
                 max_images,
                 clamp=True,
                 increase_log_steps=True,
                 rescale=True,
                 disabled=False,
                 log_on_batch_idx=False,
                 log_first_step=True,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [
            2**n for n in range(6,
                                int(np.log2(self.batch_freq)) + 1)
        ]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, prompts, global_step,
                  current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        names = {
            "reals": "cond",
            "inputs": "gt",
            "reconstruction": "recon-vq",
            "samples": "estimated"
        }
        # print(root)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=8)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(
                global_step, current_epoch, batch_idx, names[k])
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            # print(path)
            Image.fromarray(grid).save(path)

        filename = "gs-{:06}_e-{:06}_b-{:06}_prompt.json".format(
            global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        with open(path, "w") as f:
            for p in prompts:
                f.write(f"{json.dumps(p)}\n")

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx)
                and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0) or (split == "val" and batch_idx == 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch,
                                              split=split,
                                              **self.log_images_kwargs)

            prompts = batch["edit"]["c_crossattn"][:self.max_images]
            prompts = [p for ps in all_gather(prompts) for p in ps]

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                images[k] = torch.cat(all_gather(images[k][:N]))
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images, prompts,
                           pl_module.global_step, pl_module.current_epoch,
                           batch_idx)

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or
            (check_idx in self.log_steps)) and (check_idx > 0
                                                or self.log_first_step):
            if len(self.log_steps) > 0:
                self.log_steps.pop(0)
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                           dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0
                                  or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm
                    and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
