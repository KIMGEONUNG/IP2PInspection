#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ip2p
python edit_app.py --ckpt checkpoints/denoise.ckpt --config configs/generate.yaml
