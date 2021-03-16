#!/usr/bin/env bash

DATASET=$1
BASE_DIR=$2

python sched_by_conflicts_ra.py --dataset $DATASET --base_dir $BASE_DIR \
                          --alloc_file=alloc_sbc_${DATASET}