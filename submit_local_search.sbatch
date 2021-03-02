#! /bin/bash

#SBATCH --job-name=cvpr_local_search
#SBATCH --output=cvpr_local_search.out
#SBATCH --error=cvpr_local_search.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --partition=defq

OUTBASE=/mnt/nfs/scratch1/jpayan

DATASET=$1
BASE_DIR=${OUTBASE}/fair-matching/data
OUTBASE=${OUTBASE}/ReviewerAssignmentCode

./run_local_search.sh $DATASET $BASE_DIR 20