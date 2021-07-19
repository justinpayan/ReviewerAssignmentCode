# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import os
import random
import sys

from autoassigner import *
from copy import deepcopy
from utils import *
from common_fn_tests import run_probabilistic_max_exp, run_submod_exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi", type=str, default="0")
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    phi = args.phi
    iter = args.iter

    pra = np.load("mallows_data/mallows_scores_{}_{}.npy".format(phi, iter))
    covs = np.load("mallows_data/mallows_covs_{}_{}.npy".format(phi, iter)).astype(np.int)
    loads = np.load("mallows_data/mallows_loads_{}_{}.npy".format(phi, iter)).astype(np.int64)

    best_revs = np.argsort(-1 * pra, axis=0)

    # run_probabilistic_max_exp(pra, covs, loads, best_revs)
    run_submod_exp(pra, covs, loads, best_revs, norm=1, min_size=5)




