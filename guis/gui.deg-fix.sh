#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ip2p
python edit_app.py --ckpt checkpoints/deg-fix/epoch=000000-step=000000999.ckpt \
                   --config configs/generate.yaml \
                   --resolution 256
