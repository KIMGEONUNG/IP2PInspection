#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ip2p
python edit_app.py --ckpt checkpoints/openimage_s/last.ckpt \
                   --config configs/generate_highfreq.yaml \
                   --resolution 512
