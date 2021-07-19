#!/usr/bin/env bash

OUTBASE=/mnt/nfs/scratch1/jpayan

DATASET=$1
BASE_DIR=${OUTBASE}/fair-matching/data
OUTBASE=${OUTBASE}/ReviewerAssignmentCode

mkdir -p $OUTBASE/logs/tpms

sbatch -J TPMS \
          -e $OUTBASE/logs/tpms/${DATASET}.err \
          -o $OUTBASE/logs/tpms/${DATASET}.log \
          --mem=10G \
          --partition=defq \
          --time=01:00:00 \
          ./run_tpms.sh $DATASET $BASE_DIR