#!/usr/bin/env bash

CONFIG=$1

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=9996 basicsr/train.py -opt $CONFIG --launcher pytorch
