#!/bin/bash

set -e

source config.sh
source $condapath
conda activate ip2p

python main.py --name HF00 --base configs/highfreq_sd.yaml --train --gpus 0,
