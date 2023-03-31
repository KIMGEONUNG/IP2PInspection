#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ip2p
python edit_app.py --ckpt checkpoints/highfreq_sd_512/epoch=000000.ckpt \
                   --config configs/generate_highfreq.yaml \
                   --resolution 512
