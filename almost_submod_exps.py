# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import os
import random
import sys

from autoassigner import *
from copy import deepcopy
from utils import *


# Return the usw of running round robin on the agents in the list "seln_order"
def rr_usw(seln_order, pra, covs, loads, best_revs):
    # rr_alloc, rev_loads_remaining, matrix_alloc = rr(seln_order, pra, covs, loads, best_revs)
    _, rev_loads_remaining, matrix_alloc = rr(seln_order, pra, covs, loads, best_revs, output_alloc=False)
    # _usw = usw(rr_alloc, pra)
    _usw = np.sum(matrix_alloc * pra)
    # print("USW ", time.time() - start)
    return _usw, rev_loads_remaining, matrix_alloc


def rr(seln_order, pra, covs, loads, best_revs, output_alloc=True):
    if output_alloc:
        alloc = {p: list() for p in seln_order}
    matrix_alloc = np.zeros((pra.shape), dtype=np.bool)

    loads_copy = loads.copy()

    # Assume all covs are the same
    if output_alloc:
        for _ in range(covs[seln_order[0]]):
            for a in seln_order:
                for r in best_revs[:, a]:
                    if loads_copy[r] > 0 and r not in alloc[a]:
                        loads_copy[r] -= 1
                        alloc[a].append(r)
                        matrix_alloc[r, a] = 1
                        break
        return alloc, loads_copy, matrix_alloc
    else:
        for _ in range(covs[seln_order[0]]):
            for a in seln_order:
                for r in best_revs[:, a]:
                    if loads_copy[r] > 0 and matrix_alloc[r, a] == 0:
                        loads_copy[r] -= 1
                        matrix_alloc[r, a] = 1
                        break

        return None, loads_copy, matrix_alloc


def run_submod_exp(pra, covs, loads, best_revs):
    m, n = pra.shape
    diffs = []

    for i in range(10000):
        if i % 100 == 0 and i > 0:
            print(i)
            d = np.array(diffs)
            print(np.sum(d > 1e-8))
            print(np.mean(d > 1e-8) * 100)
            print(np.mean(d[d > 1e-8]))
            print(np.std(d[d > 1e-8]))
            print()
        # Determine a big set of agents Y, and then sample from it to get the littler set X.
        # Select a single element e.
        # Randomly order Y + e, and assign each element a number based on the ordering.
        # Now use those numbers to order Y + e, Y, X + e, and X.
        # Compute rr_usw for all of these.

        agents_Ye = np.random.randint(0, 2, (n))
        e = np.random.choice(np.where(agents_Ye)[0])
        agents_Y = agents_Ye.copy()
        agents_Y[e] = 0
        agents_Xe = np.random.randint(0, 2, (n)) * agents_Y
        agents_Xe[e] = 1
        agents_X = agents_Xe.copy()
        agents_X[e] = 0

        agents_Ye = np.where(agents_Ye)[0].tolist()
        agents_Y = np.where(agents_Y)[0].tolist()
        agents_Xe = np.where(agents_Xe)[0].tolist()
        agents_X = np.where(agents_X)[0].tolist()

        total_order = sorted(agents_Ye, key=lambda x: random.random())
        order_map = {i: idx for idx, i in enumerate(total_order)}

        agents_Ye = sorted(agents_Ye, key=lambda x: order_map[x])
        agents_Y = sorted(agents_Y, key=lambda x: order_map[x])
        agents_Xe = sorted(agents_Xe, key=lambda x: order_map[x])
        agents_X = sorted(agents_X, key=lambda x: order_map[x])

        diff_y = rr_usw(agents_Ye, pra, covs, loads, best_revs)[0] - rr_usw(agents_Y, pra, covs, loads, best_revs)[0]
        diff_x = rr_usw(agents_Xe, pra, covs, loads, best_revs)[0] - rr_usw(agents_X, pra, covs, loads, best_revs)[0]

        diffs.append(diff_y - diff_x)

    return np.array(diffs)


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

    diffs = run_submod_exp(paper_reviewer_affinities, covs, loads, best_revs)
    print(diffs)
    print(np.sum(diffs > 1e-8))
    print(np.mean(diffs > 1e-8)*100)
    print(np.mean(diffs[diffs > 1e-8]))
    print(np.std(diffs[diffs > 1e-8]))
    # print(np.mean(diffs))
    # print(np.std(diffs))



