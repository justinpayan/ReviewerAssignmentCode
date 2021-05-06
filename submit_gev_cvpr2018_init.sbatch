#! /bin/bash

NUM_DISTRIB_JOBS=1000
NUM_PROCESSES=40

DATASET=cvpr2018

OUTBASE=/mnt/nfs/scratch1/jpayan

BASE_DIR=${OUTBASE}/fair-matching/data
OUTBASE=${OUTBASE}/ReviewerAssignmentCode

for JOB_NUM in {0..$NUM_DISTRIB_JOBS}; do
    sbatch -J cvpr2018_gev_init_${JOB_NUM} \
           -e cvpr2018_gev_init_${JOB_NUM}.err \
           -o cvpr2018_gev_init_${JOB_NUM}.out \
           --mem=40G \
           --time=12:00:00 \
           --partition=defq \
           --nodes=1 \
           --ntasks=40 \
            ./run_gev_init.sh $DATASET $BASE_DIR $NUM_PROCESSES $JOB_NUM $NUM_DISTRIBUTED_JOB
done