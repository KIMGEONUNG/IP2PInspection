import numpy as np
import cv2
import math
import random
import utils.degradations as degradations
from random import sample
from PIL import Image


class Degrade(object):

    def __init__(self):
        self.kernel_list = ['iso', 'aniso']
        self.kernel_prob = [0.5, 0.5]
        self.blur_kernel_size = 41
        self.blur_sigma = [0.1, 10]
        self.downsample_range = [0.8, 8]
        self.noise_range = [0, 100]
        self.jpeg_range = [60, 100]
        self.gray_prob = 0.2
        self.color_jitter_prob = 0.0
        self.color_jitter_pt_prob = 0.0
        self.shift = 20 / 255.

        self.targets = ["blur", "gray", "noise", "downsample"]
        self.intact_prop = 0.05

    def degrade_process(self, img_gt):
        h, w = img_gt.shape[:2]

        # random grayscale
        if np.random.uniform() < self.gray_prob and self.use_gray:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(self.kernel_list,
                                                   self.kernel_prob,
                                                   self.blur_kernel_size,
                                                   self.blur_sigma,
                                                   self.blur_sigma,
                                                   [-math.pi, math.pi],
                                                   noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0],
                                  self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)),
                            interpolation=cv2.INTER_LINEAR)

        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(
                img_lq, self.noise_range)

        # round and clip
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        return img_gt, img_lq

    def random_single_deg(self, img: Image):
        target = sample(self.targets, 1)[0]

        img_gt = np.asarray(img)
        img_gt = img_gt.astype(np.float32) / 255.

        h, w = img_gt.shape[:2]
        img_lq = img_gt

        prompt = ""
        if target == "blur":
            prompt += "deblur"
            kernel = degradations.random_mixed_kernels(self.kernel_list,
                                                       self.kernel_prob,
                                                       self.blur_kernel_size,
                                                       self.blur_sigma,
                                                       self.blur_sigma,
                                                       [-math.pi, math.pi],
                                                       noise_range=None)
            img_lq = cv2.filter2D(img_lq, -1, kernel)

        if target == "gray":
            prompt += "colorize"
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])

        if target == "noise":
            prompt += "denoise"
            img_lq = degradations.random_add_gaussian_noise(
                img_lq, self.noise_range)

        if target == "downsample":
            scale = np.random.uniform(self.downsample_range[0],
                                      self.downsample_range[1])
            img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)),
                                interpolation=cv2.INTER_LINEAR)
            prompt += "make it full HD"

        # round and clip
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        if np.random.uniform() < self.intact_prop:
            img_lq = img_gt

        img_lq = (img_lq * 255).astype(np.uint8)
        img_lq = Image.fromarray(img_lq)

        return img_lq, prompt


if __name__ == "__main__":
    from skimage.io import imread
    from tqdm import tqdm
    from PIL import Image

    degrader = Degrade()
    x = Image.open('sample01.jpg')

    for i in tqdm(range(100)):
        y, t = degrader.random_single_deg(x)
        t = t.replace(' ', '_')
        y.save('tmp_a/%05d_%s.jpg' % (i, t))
