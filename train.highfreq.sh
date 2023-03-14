#!/bin/bash

source config.sh
source $condapath
conda activate ip2p

python main.py --name HF --base configs/highfreq_sd.yaml --train --gpus 0,1
