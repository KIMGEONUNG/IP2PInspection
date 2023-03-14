#!/bin/bash

set -e

source config.sh
source $condapath
conda activate ip2p

id=$(date +%Y%m%d-%H%M%S)

python main.py --name HF_$id --base configs/highfreq_sd.yaml --train --gpus 0,
