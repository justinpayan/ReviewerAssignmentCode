#! /bin/bash

#SBATCH --job-name=midl_lagrangian_relax
#SBATCH --output=midl_lagrangian_relax.out
#SBATCH --error=midl_lagrangian_relax.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --time=06-00:00:00
#SBATCH --partition=longq

OUTBASE=/mnt/nfs/scratch1/jpayan

DATASET=$1
BASE_DIR=${OUTBASE}/fair-matching/data
OUTBASE=${OUTBASE}/ReviewerAssignmentCode

./run_lag_rel.sh $DATASET $BASE_DIR