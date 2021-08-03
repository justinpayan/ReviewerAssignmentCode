# Optimizing Orders for Fair Reviewer Assignment

This repository implements the Greedy Reviewer Round Robin reviewer assignment algorithm from our paper, I Will Have Order! Optimizing Orders for Fair Reviewer Assignment. All analysis and baselines can be run using code in this repository as well.
Please cite our paper if you use this work.

## Data

The data for all three conferences can be obtained from the GitHub repository for [Paper Matching with Local Fairness Constraints](https://github.com/iesl/fair-matching).

Clone the above repository in the top level of this repository, and run `unzip fair-matching/data/cvpr2018/scores.zip`.

## Experiments

We show how to run all experiments and baselines on the MIDL conference. All algorithms/analysis can be run on CVPR and CVPR2018 by replacing `--dataset midl` with `--dataset cvpr` or `--dataset cvpr2018` in any command. 

### Greedy Reviewer Round Robin

To run GRRR for the MIDL dataset, run

`python greedy_reviewer_round_robin.py --dataset midl --alloc_file midl_grrr`

This will save the final allocation in `midl_grrr`, and it will save the order on papers in `midl_grrr_order`. In addition, it will also print out all the statistics reported in the paper.

The file `estimate_alpha_gamma.py` can be used to estimate the values alpha and gamma as described in our paper, by running

`python estimate_alpha_gamma.py --dataset midl`

By default, this will estimate gamma using the alpha values mentioned in the paper. To estimate alpha, comment out the final line of the file and uncomment the line above it running the method `estimate_alpha`. 

### Baselines

All baselines require the [gurobi](https://www.gurobi.com) library, which can be installed with a free academic license.

To evaluate FairFlow and FairIR using our metrics, first execute them both on the desired dataset following the directions in that repository. The allocations will appear in timestamped directories under `fair-matching/exp_out`. Then run

`python evaluate_fairflow_fairir.py --dataset midl --fairflow_timestamp <TIMESTAMP> --fairir_timestamp <TIMESTAMP>`

To execute and compute statistics for PeerReview4All, run `python pr4a_wrapper.py --dataset midl --alloc_file midl_pr4a`. This will call the code in `autoassigner.py`, which we obtained from [Ivan Stelmakh's website](https://www.cs.cmu.edu/~istelmak/) and minimally modified to incorporate arbitrary reviewer load upper bounds.

To run and evaluate CRR, run `python constrained_rr.py --dataset midl --alloc_file midl_crr --w_value 1.68`. The `w_value` parameter specifies the mean welfare target, so the algorithm will return an allocation with average welfare at least that high (if possible).

To run and evaluate TPMS, run `python tpms.py --dataset midl --alloc_file midl_tpms`.

## Contact

Please contact Justin Payan (`jpayan@umass.edu`) for any questions/comments/discussion about the code or the paper.