#!/usr/bin/env bash

DATASET=$1
BASE_DIR=$2

python sgd_lagrangian_relaxation.py --dataset $DATASET --base_dir $BASE_DIR \
                          --alloc_file=lagrangian_relaxation_${DATASET}