#!/usr/bin/env bash

module load gurobi/1001

I=$1

python pr4a_wrapper.py --base_dir /mnt/nfs/scratch1/jpayan/ReviewerAssignmentCode/data \
  --dataset ijcai23 \
  --alloc_file pr4a_alloc_ijcai_$I