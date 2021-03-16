#! /bin/bash

#SBATCH --job-name=cvpr_sbc
#SBATCH --output=cvpr_sbc.out
#SBATCH --error=cvpr_sbc.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --partition=defq

OUTBASE=/mnt/nfs/scratch1/jpayan

DATASET=$1
BASE_DIR=${OUTBASE}/fair-matching/data
OUTBASE=${OUTBASE}/ReviewerAssignmentCode

./run_sched_by_conflicts.sh $DATASET $BASE_DIR