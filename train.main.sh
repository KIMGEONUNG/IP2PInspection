#!/bin/bash

set -e

source config.sh
source $condapath
conda activate ip2p

id=$(date +%Y%m%d-%H%M%S)

if [[ -z $1 ]]; then
    echo -e "\033[31mError: no config arg \033[0m" >&2
    echo -e "\033[31mHint: configs/highfreq_sd_512.yaml \033[0m" >&2
    exit 0
fi

python main.py --name $id --base $1 --train --gpus $id_gpu
