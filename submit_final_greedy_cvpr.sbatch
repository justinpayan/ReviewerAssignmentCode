#! /bin/bash

#SBATCH --job-name=cvpr_final
#SBATCH --output=cvpr_final.out
#SBATCH --error=cvpr_final.err
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem=40G
#SBATCH --time=06-00:00:00
#SBATCH --partition=longq

DATASET=cvpr

OUTBASE=/mnt/nfs/scratch1/jpayan

BASE_DIR=${OUTBASE}/fair-matching/data
OUTBASE=${OUTBASE}/ReviewerAssignmentCode

./run_final_greedy.sh $DATASET $BASE_DIR 40
