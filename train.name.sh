#!/bin/bash

set -e

source config.sh
source $condapath
conda activate ip2p

id=$(date +%Y%m%d-%H%M%S)

if [[ -z $1 ]]; then
    echo -e "\033[31mError: no name arg \033[0m" >&2
    echo -e "\033[31mHint: T001-A00 \033[0m" >&2
    exit 0
fi

python ./exps/${1}.py --name $id --train --gpus $id_gpu
