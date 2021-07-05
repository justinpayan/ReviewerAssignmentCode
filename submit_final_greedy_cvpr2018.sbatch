#! /bin/bash

seed=$1

#SBATCH --job-name=cvpr2018_final_${seed}
#SBATCH --output=cvpr2018_final_${seed}.out
#SBATCH --error=cvpr2018_final_${seed}.err
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem=40G
#SBATCH --time=06-00:00:00
#SBATCH --partition=longq

DATASET=cvpr2018

OUTBASE=/mnt/nfs/scratch1/jpayan

BASE_DIR=${OUTBASE}/fair-matching/data
OUTBASE=${OUTBASE}/ReviewerAssignmentCode

./run_final_greedy.sh $DATASET $BASE_DIR 40 $seed
