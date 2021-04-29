#! /bin/bash

#SBATCH --job-name=midl_gev
#SBATCH --output=midl_gev.out
#SBATCH --error=midl_gev.err
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem=40G
#SBATCH --time=06-00:00:00
#SBATCH --partition=longq

DATASET=midl

OUTBASE=/mnt/nfs/scratch1/jpayan

BASE_DIR=${OUTBASE}/fair-matching/data
OUTBASE=${OUTBASE}/ReviewerAssignmentCode

./run_gev.sh $DATASET $BASE_DIR 40
