#! /bin/bash

#SBATCH --job-name=cvpr2018_gev
#SBATCH --output=cvpr2018_gev.out
#SBATCH --error=cvpr2018_gev.err
#SBATCH --nodes=1
#SBATCH --ntasks=500
#SBATCH --mem=40G
#SBATCH --time=06-00:00:00
#SBATCH --partition=longq

DATASET=cvpr2018

OUTBASE=/mnt/nfs/scratch1/jpayan

BASE_DIR=${OUTBASE}/fair-matching/data
OUTBASE=${OUTBASE}/ReviewerAssignmentCode

./run_gev.sh $DATASET $BASE_DIR 40
