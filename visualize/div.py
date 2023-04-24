#!/usr/bin/env python

from PIL import Image
import numpy as np
import argparse
from glob import glob
from os.path import join


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--path', default='train')
    p.add_argument('--div', type=int, default=8)
    return p.parse_args()


def main():
    args = parse()
    for p in glob(join(args.path, '*.png')):
        img = Image.open(p)
        x = np.array(img)
        unit = x.shape[1] // args.div
        p_wo_ext = p.split('.')[0]
        for i in range(args.div):
            x_ = x[:, i * unit:(i + 1) * unit, :]
            p_out = p_wo_ext + f"_div_{i:02}.png"
            img_ = Image.fromarray(x_)
            img_.save(p_out)

if __name__ == "__main__":
    main()
