import math
import multiprocessing as mp
import os
import random
import time

from greedy_rr import safe_rr_usw, safe_rr, _greedy_rr_ordering
from utils import *


def compute_marginal_gain(input_args):
    pair, tuple_set, current_usw, scores, covs, loads, best_revs, n_iters, norm = input_args
    usw = estimate_expected_safe_rr_usw(tuple_set | {pair}, scores, covs, loads, best_revs,
                                        n_iters=n_iters, normalizer=norm)
    mg = usw - current_usw
    return mg


"""Approximately compute the set of tuples corresponding to a total ordering which maximizes the USW from Reviewer RR,
by running greedy and every time we violate submodularity reduce the marginal contribution so that we don't."""


def greedy_based_on_ev(scores, loads, covs, best_revs, n_iters, norm, num_processes):
    m, n = scores.shape

    available_agents = set(range(n))
    available_positions = set(range(n))

    pool = mp.Pool(processes=num_processes)

    tuple_set = set()
    marginal_gains = {p: np.inf for p in product(available_agents, available_positions)}
    current_usw = 0
    for _ in range(n):
        pair_to_add = None
        best_marginal_gain = -1

        if len(available_agents)*len(available_positions) < 1000:
            for pair in product(available_agents, available_positions):
                # We can skip if the upper bound on the marginal gain (which is basically an ENFORCED upper bound)
                # is not enough.
                if marginal_gains[pair] > best_marginal_gain:
                    mg = compute_marginal_gain((pair, tuple_set, current_usw, scores, covs, loads, best_revs, n_iters, norm))
                    mg = min([mg, marginal_gains[pair]])
                    marginal_gains[pair] = mg

                    if mg > best_marginal_gain:
                        best_marginal_gain = mg
                        pair_to_add = pair
        else:
            for idx, a in enumerate(available_agents):
                if idx % 5 == 0:
                    print("idx: {}, best_mg_so_far: {}, pair_to_add: {}".format(idx, best_marginal_gain, pair_to_add),
                          flush=True)

                pairs_to_try = [(a, p) for p in available_positions if marginal_gains[(a, p)] > best_marginal_gain]
                print("len(pairs_to_try): {}".format(len(pairs_to_try)))

                list_of_copied_args = [pairs_to_try]
                for argument in [tuple_set, current_usw, scores, covs, loads, best_revs, n_iters, norm]:
                    list_of_copied_args.append(len(pairs_to_try) * [argument])

                # print(list(zip(*list_of_copied_args))[0])

                mgs = pool.map(compute_marginal_gain, zip(*list_of_copied_args))

                for p, mg in zip(pairs_to_try, mgs):
                    mg = min([mg, marginal_gains[p]])
                    marginal_gains[p] = mg

                    if mg > best_marginal_gain:
                        best_marginal_gain = mg
                        pair_to_add = p

        tuple_set.add(pair_to_add)
        current_usw += marginal_gains[pair_to_add]
        available_agents.remove(pair_to_add[0])
        available_positions.remove(pair_to_add[1])

    return tuple_set


def run_algo(dataset, base_dir, num_processes, alloc_file):
    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)

    norm_map = {"midl": 201.88, "cvpr": 5443.64, "cvpr2018": 112552.11}

    best_revs = np.argsort(-1 * scores, axis=0)

    tuple_set = greedy_based_on_ev(scores, loads, covs, best_revs, 5, norm_map[dataset], num_processes)
    ordering = [x[0] for x in sorted(tuple_set, key=lambda x: x[1])]
    alloc = safe_rr(ordering, scores, covs, loads, best_revs)[0]
    return alloc, ordering


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = args.base_dir
    alloc_file = args.alloc_file
    num_processes = args.num_processes

    random.seed(args.seed)

    start = time.time()
    alloc, ordering = run_algo(dataset, base_dir, num_processes, alloc_file)
    runtime = time.time() - start

    save_alloc(alloc, alloc_file)
    save_alloc(ordering, alloc_file + "_order")

    # print("Barman Algorithm Results")
    # print("%.2f seconds" % runtime)
    # print_stats(alloc, paper_reviewer_affinities, paper_capacities)

    print("Corrected Greedy by Expected Value Results")
    print("%.2f seconds" % runtime)

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy"))

    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, paper_capacities)
