#! /bin/bash

DATASET=$1
ALG=$2

for I in {0..9}; do
  python speed_tests.py --dataset $DATASET --data_dir /mnt/nfs/scratch1/jpayan/fair-matching/data --algorithm $ALG --iter $I
done

