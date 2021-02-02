#!/usr/bin/env bash

DATASET=$1
BASE_DIR=$2

python gurobi_usw_ef1.py --dataset $DATASET --base_dir $BASE_DIR