#! /bin/bash

DATASET=$1

#SBATCH --job-name=${DATASET}_local_search
#SBATCH --output=${DATASET}_local_search.out
#SBATCH --error=${DATASET}_local_search.err
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem=40G
#SBATCH --time=06-00:00:00
#SBATCH --partition=longq

OUTBASE=/mnt/nfs/scratch1/jpayan

BASE_DIR=${OUTBASE}/fair-matching/data
OUTBASE=${OUTBASE}/ReviewerAssignmentCode

./run_local_search.sh $DATASET $BASE_DIR 40