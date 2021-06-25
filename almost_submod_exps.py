# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import os
import random
import sys

from autoassigner import *
from copy import deepcopy
from utils import *
from common_fn_tests import run_submod_exp


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    dataset = args.dataset
    alloc_file = args.alloc_file
    seed = args.seed

    np.random.seed(seed)
    random.seed(seed)

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)

    if np.max(covs) != np.min(covs):
        print("covs must be homogenous")
        sys.exit(1)

    # greedy_rr(paper_reviewer_affinities, covs, loads, args.alloc_file)
    best_revs = np.argsort(-1 * paper_reviewer_affinities, axis=0)

    norm_map = {"midl": 201.88, "cvpr": 5443.64, "cvpr2018": 112552.11}
    norm = norm_map[dataset]

    min_size = 5

    run_submod_exp(paper_reviewer_affinities, covs, loads, best_revs, norm, min_size)
    # run_probabilistic_max_exp(paper_reviewer_affinities, covs, loads, best_revs, norm)
    # print(diffs)
    # print(np.sum(diffs > 1e-8))
    # print(np.mean(diffs > 1e-8)*100)
    # print(np.mean(diffs[diffs > 1e-8]))
    # print(np.std(diffs[diffs > 1e-8]))
    # print(np.mean(diffs))
    # print(np.std(diffs))



