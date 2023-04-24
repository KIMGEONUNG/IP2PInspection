#!/bin/bash

path=${path:-train}
if [[ -z $path ]]; then
  echo no path
  exit 0
fi

path_root=${path}_g
path_cond=${path_root}/a_cond
path_est=${path_root}/b_est
path_gt=${path_root}/c_gt
path_recon=${path_root}/d_recon

mkdir -p ${path_root}
mkdir -p ${path_cond}
mkdir -p ${path_est}
mkdir -p ${path_gt}
mkdir -p ${path_recon}

for p in ${path}/*cond*; do
  cp -v $p $path_cond
done

for p in ${path}/*gt*; do
  cp -v $p $path_gt
done

for p in ${path}/*est*; do
  cp -v $p $path_est
  echo $p
done

for p in ${path}/*recon*; do
  cp -v $p $path_recon
  echo $p
done
