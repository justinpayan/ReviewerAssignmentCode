import math
import multiprocessing as mp
import os
import random
import time

from pathlib import Path

from greedy_rr import safe_rr_usw, safe_rr, _greedy_rr_ordering
from utils import *


def compute_init_marginal_gain(input_args):
    pair, current_usw, scores, covs, loads, best_revs, n_iters, norm = input_args
    usw = estimate_expected_safe_rr_usw({pair}, scores, covs, loads, best_revs,
                                        n_iters=n_iters, normalizer=norm)
    mg = usw - current_usw
    return mg


def compute_marginal_gain(input_args):
    pair, tuple_set, current_usw, scores, covs, loads, best_revs, n_iters, norm = input_args
    usw = estimate_expected_safe_rr_usw(tuple_set | {pair}, scores, covs, loads, best_revs,
                                        n_iters=n_iters, normalizer=norm)
    mg = usw - current_usw
    return mg


"""Approximately compute the set of tuples corresponding to a total ordering which maximizes the USW from Reviewer RR,
by running greedy and every time we violate submodularity reduce the marginal contribution so that we don't."""


def greedy_based_on_ev(scores, loads, covs, best_revs, n_iters, norm, num_processes, initial_marginal_gains):
    m, n = scores.shape

    available_agents = set(range(n))
    available_positions = set(range(n))

    pool = mp.Pool(processes=num_processes)

    tuple_set = set()
    marginal_gains = {p: np.inf for p in product(available_agents, available_positions)}
    current_usw = 0

    if initial_marginal_gains:
        marginal_gains = initial_marginal_gains

        pair_to_add = None
        best_marginal_gain = -1

        for pair in product(available_agents, available_positions):
            # We can skip if the upper bound on the marginal gain (which is basically an ENFORCED upper bound)
            # is not enough.
            if marginal_gains[pair] > best_marginal_gain:
                best_marginal_gain = marginal_gains[pair]
                pair_to_add = pair

        tuple_set.add(pair_to_add)
        current_usw += marginal_gains[pair_to_add]
        available_agents.remove(pair_to_add[0])
        available_positions.remove(pair_to_add[1])

    while len(tuple_set) < n:
        pair_to_add = None
        best_marginal_gain = -1

        if len(available_agents)*len(available_positions) < 1000:
            for pair in product(available_agents, available_positions):
                # We can skip if the upper bound on the marginal gain (which is basically an ENFORCED upper bound)
                # is not enough.
                if marginal_gains[pair] > best_marginal_gain:
                    mg = compute_marginal_gain((pair, tuple_set, current_usw,
                                                scores, covs, loads, best_revs, n_iters, norm))
                    mg = min([mg, marginal_gains[pair]])
                    marginal_gains[pair] = mg

                    if mg > best_marginal_gain:
                        best_marginal_gain = mg
                        pair_to_add = pair
        else:
            pairs_to_try = [(a, p) for (a, p) in product(available_agents, available_positions)
                            if marginal_gains[(a, p)] > best_marginal_gain]

            print("len(pairs_to_try): {}".format(len(pairs_to_try)))
            print("len(tuple_set): {}".format(len(tuple_set)))
            print("current_usw: {}".format(current_usw), flush=True)

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


def try_to_merge(mg_file, num_distrib_jobs, n):
    print("Trying to merge all %d marginal gain files" % num_distrib_jobs)

    for i in range(num_distrib_jobs):
        if not Path("{}_{}_{}".format(mg_file, i, num_distrib_jobs)).is_file():
            print("Failed to merge: File %d of %d not complete" % (i, num_distrib_jobs), flush=True)
            return

    marginal_gains = {p: np.inf for p in product(set(range(n)), set(range(n)))}

    for i in range(num_distrib_jobs):
        print("Merging file %d" % i, flush=True)
        with open("{}_{}_{}".format(mg_file, i, num_distrib_jobs), 'rb') as f:
            local_mg = pickle.load(f)
            for x in local_mg:
                marginal_gains[x] = min(marginal_gains[x], local_mg[x])

    print("Saving final marginal gain file", flush=True)
    with open(mg_file, 'wb') as f:
        pickle.dump(marginal_gains, f)


def init_mg_files(dataset, base_dir, n_iters, norm, num_processes, mg_file, num_distrib_jobs, job_num):
    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)

    best_revs = np.argsort(-1 * scores, axis=0)

    m, n = scores.shape

    agents_this_job = {i for i in range(n) if i % num_distrib_jobs == job_num}

    available_agents = agents_this_job
    available_positions = set(range(n))

    pool = mp.Pool(processes=num_processes)

    marginal_gains = {p: np.inf for p in product(available_agents, available_positions)}
    current_usw = 0

    pairs_to_try = [(a, p) for (a, p) in product(available_agents, available_positions)]

    print("len(pairs_to_try): {}".format(len(pairs_to_try)), flush=True)

    list_of_copied_args = [pairs_to_try]
    for argument in [current_usw, scores, covs, loads, best_revs, n_iters, norm]:
        list_of_copied_args.append(len(pairs_to_try) * [argument])

    # print(list(zip(*list_of_copied_args))[0])

    mgs = pool.map(compute_init_marginal_gain, zip(*list_of_copied_args))

    for p, mg in zip(pairs_to_try, mgs):
        marginal_gains[p] = mg

    with open("{}_{}_{}".format(mg_file, job_num, num_distrib_jobs), 'wb') as f:
        pickle.dump(marginal_gains, f)

    # Check if they are all finished, and if so, merge all the files
    try_to_merge(mg_file, num_distrib_jobs, n)


def run_algo(dataset, base_dir, n_iters, norm, num_processes, alloc_file, mg_file=None):
    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)

    best_revs = np.argsort(-1 * scores, axis=0)

    initial_marginal_gains = None
    if mg_file:
        with open(mg_file, 'rb') as f:
            initial_marginal_gains = pickle.load(f)

    tuple_set = greedy_based_on_ev(scores, loads, covs, best_revs, n_iters,
                                   norm, num_processes, initial_marginal_gains)
    ordering = [x[0] for x in sorted(tuple_set, key=lambda x: x[1])]
    alloc = safe_rr(ordering, scores, covs, loads, best_revs)[0]
    return alloc, ordering


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = args.base_dir
    alloc_file = args.alloc_file
    num_processes = args.num_processes
    mg_file = args.mg_file
    init_run = args.init_run
    num_distrib_jobs = args.num_distrib_jobs
    job_num = args.job_num

    random.seed(args.seed)

    norm_map = {"midl": 201.88, "cvpr": 5443.64, "cvpr2018": 112552.11}
    norm = norm_map[dataset]
    n_iters = 5

    if mg_file and init_run:
        init_mg_files(dataset, base_dir, n_iters, norm, num_processes, mg_file, num_distrib_jobs, job_num)
    else:
        start = time.time()
        alloc, ordering = run_algo(dataset, base_dir, n_iters, norm, num_processes, alloc_file, mg_file)
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
