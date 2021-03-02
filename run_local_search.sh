#!/usr/bin/env bash

DATASET=$1
BASE_DIR=$2
NUM_PROCESSES=$3

python local_search_rr.py --dataset $DATASET --base_dir $BASE_DIR \
                          --num_processes=$NUM_PROCESSES \
                          --alloc_file=local_search_${DATASET}_partial_with_greedy_init \
                          --local_search_init_order partial_order_cvpr_debug