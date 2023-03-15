#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ip2p
python edit_app.py --ckpt checkpoints/instruct-pix2pix-00-22000.ckpt --config configs/generate.yaml
