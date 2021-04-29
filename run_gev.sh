#!/usr/bin/env bash

DATASET=$1
BASE_DIR=$2
NUM_PROCESSES=$3

python greedy_based_on_expected_value.py --dataset $DATASET --base_dir $BASE_DIR \
                                         --num_processes $NUM_PROCESSES \
                                         --alloc_file gev_${DATASET}