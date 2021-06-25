#!/usr/bin/env bash

DATASET=$1
BASE_DIR=$2
NUM_PROCESSES=$3

python final_greedy_algo.py --dataset $DATASET --base_dir $BASE_DIR \
                                         --num_processes $NUM_PROCESSES \
                                         --alloc_file final_greedy_${DATASET}