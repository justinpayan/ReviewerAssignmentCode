#! /bin/bash

for PHI in "0" "0.25" "0.5" "0.75" "0.95" do
    for ITER in {0..4}; do
        sbatch -J mmt_${PHI}_${ITER} \
        ./run_mallows_max_test.sh $PHI $ITER
    done
done