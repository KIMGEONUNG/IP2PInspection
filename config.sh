#!/bin/bash

if [[ $(hostname | grep mark12) ]]; then
    export condapath=/opt/conda/etc/profile.d/conda.sh
    # export CUDA_VISIBLE_DEVICES=4,5
    export CUDA_VISIBLE_DEVICES=5
    # if ! [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    # fi
else
    export condapath=$HOME/anaconda3/etc/profile.d/conda.sh
    if ! [[ -z $CUDA_VISIBLE_DEVICES ]]; then
        export CUDA_VISIBLE_DEVICES=0
    fi
fi
