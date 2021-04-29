# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import os
import random
import sys

from autoassigner import *
from copy import deepcopy
from utils import *


def run_submod_exp(pra, covs, loads, best_revs):
    m, n = pra.shape
    diffs = []
    monotonic = []
    c_values = []

    # Sample a set of tuples for X, then sample e1 and e2. Run expected safe rr_usw for X, X+e1, X+e2,
    #  and X+e1+e2. Record amount of violation of submodularity, and if monotonicity was violated.
    for num_agents in [n-2, n-3, n-4, n-5] + list(range(50, n, 1000)) :
        for _ in range(5):
            agents_X_e1_e2 = random.sample(list(range(n)), num_agents)
            positions = random.sample(list(range(n)), num_agents)

            X_e1_e2 = set(zip(agents_X_e1_e2, positions))
            e1 = random.sample(X_e1_e2, 1)[0]
            X_e2 = X_e1_e2 - {e1}
            e2 = random.sample(X_e2, 1)[0]
            X = X_e2 - {e2}
            X_e1 = deepcopy(X)
            X_e1.add(e1)

            # print("X: ", sorted(X, key=lambda x: x[1]))
            # print("e1: ", e1)
            # print("e2: ", e2)
            # print("X_e1: ", sorted(X_e1, key=lambda x: x[1]))
            # print("X_e2: ", sorted(X_e2, key=lambda x: x[1]))
            # print("X_e1_e2: ", sorted(X_e1_e2, key=lambda x: x[1]))

            n_iters = 10

            # empirically determined constants for MIDL, CVPR, CVPR18, in that order:
            # normalizers: 201.88, 5443.64, 112552.11 - the TPMS scores for each conference. Very easy to compute.
            # c values: >= 1.2511244577594005, ?, ? - computed by sampling 5 X + e1 + e2 sets for each |X|, step by 10.

            norm = 201.88
            # c_val = 1.26

            # norm = 5443.61
            # norm=112552.11
            # c_val = 1.5

            usw_x_e1_e2 = estimate_expected_safe_rr_usw(X_e1_e2, pra, covs, loads, best_revs, n_iters=n_iters, normalizer=norm)
            usw_x_e2 = estimate_expected_safe_rr_usw(X_e2, pra, covs, loads, best_revs, n_iters=n_iters, normalizer=norm)
            usw_x_e1 = estimate_expected_safe_rr_usw(X_e1, pra, covs, loads, best_revs, n_iters=n_iters, normalizer=norm)
            usw_x = estimate_expected_safe_rr_usw(X, pra, covs, loads, best_revs, n_iters=n_iters, normalizer=norm)

            print(usw_x_e1_e2, usw_x_e1, usw_x_e2, usw_x)

            diff_rhs = usw_x_e1_e2 - usw_x_e2
            diff_lhs = usw_x_e1 - usw_x

            diffs.append(diff_rhs - diff_lhs)
            monotonic.append(usw_x <= usw_x_e1 <= usw_x_e1_e2 and usw_x <= usw_x_e2 <= usw_x_e1_e2)
            c_values.append(usw_x_e1_e2/usw_x)

        print("|X|: ", num_agents)
        d = np.array(diffs)
        # print(np.sum(d > 1e-8))
        print(diffs)
        print("percentage of supermod viols: ", np.mean(d < 0) * 100)  # Percentage of violations
        # print(np.mean(d[d > 1e-8]))  # Mean amount of violation
        # print(np.std(d[d > 1e-8]))
        print("percent monotonic: ", np.mean(monotonic))
        print("max c: ", np.max(c_values))
        print()

    return np.array(diffs)


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    dataset = args.dataset
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

    diffs = run_submod_exp(paper_reviewer_affinities, covs, loads, best_revs)
    # print(diffs)
    # print(np.sum(diffs > 1e-8))
    # print(np.mean(diffs > 1e-8)*100)
    # print(np.mean(diffs[diffs > 1e-8]))
    # print(np.std(diffs[diffs > 1e-8]))
    # print(np.mean(diffs))
    # print(np.std(diffs))



