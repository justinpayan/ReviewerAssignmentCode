#! /bin/bash

LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/FairSequence

for I in {0..9}; do
  sbatch --time=00-11:00:00 --partition=defq \
  --nodes=1 --ntasks=1 --mem=30G --output=$LOG_DIR/pr4a_timing_test_${I}.out \
  --error=$LOG_DIR/pr4a_timing_test_${I}.err --job-name timing_${I} \
  ./speed_test_pr4a.sh
done