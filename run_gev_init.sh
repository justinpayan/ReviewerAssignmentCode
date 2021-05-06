#!/usr/bin/env bash

DATASET=$1
BASE_DIR=$2
NUM_PROCESSES=$3
JOB_NUM=$4
NUM_DISTRIB_JOBS=$5

python greedy_based_on_expected_value.py --dataset $DATASET --base_dir $BASE_DIR \
                                         --num_processes $NUM_PROCESSES \
                                         --alloc_file gev_${DATASET} \
                                         --job_num $JOB_NUM \
                                         --num_distrib_jobs $NUM_DISTRIB_JOBS \
                                         --mg_file mg_init_${DATASET} \
                                         --init_run