# Find a problem which does not admit a complete, ef1 allocation.

import os
import random
import sys

from autoassigner import *
from copy import deepcopy
from utils import *
# from common_fn_tests import run_probabilistic_max_exp, run_submod_exp
from final_greedy_algo import greedy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi", type=str, default="0")
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--alloc_file", type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    phi = args.phi
    iter = args.iter
    alloc_file = args.alloc_file

    # scores = np.load("mallows_data/mallows_scores_{}_{}.npy".format(phi, iter))
    # covs = np.load("mallows_data/mallows_covs_{}_{}.npy".format(phi, iter)).astype(np.int)
    # loads = np.load("mallows_data/mallows_loads_{}_{}.npy".format(phi, iter)).astype(np.int64)

    base_dir = "/home/justinspayan/Fall_2020/fair-matching/data"
    dataset = "cvpr"
    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))

    best_revs = np.argsort(-1 * scores, axis=0)

    covs += 1
    loads -= 2

    # print(scores)
    # print(covs)
    # print(loads)

    alloc, ordering = greedy(scores, loads, covs, best_revs, alloc_file)

    print(alloc)
    print(ef1_violations(alloc, scores))

    for p in alloc:
        if len(alloc[p]) != covs[p]:
            print(p)
            sys.exit(0)
    # run_probabilistic_max_exp(pra, covs, loads, best_revs)
    # run_submod_exp(pra, covs, loads, best_revs, norm=1, min_size=5)




