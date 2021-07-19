# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.
import math
import os
import random
import tqdm
import sys

from autoassigner import *
from copy import deepcopy
import networkx as nx

from greedy_rr import rr_usw, rr
from utils import *


def matrix_from_alloc(alloc, scores):
    matrix_alloc = np.zeros(scores.shape)
    for a, revs in alloc.items():
        for r in revs:
            matrix_alloc[r, a] = 1
    return matrix_alloc


def partial_order_to_list(order):
    # best_revs = np.argsort(-1 * scores, axis=0)
    o = sorted(order, key=lambda x: x[1])
    o = [t[0] for t in o]
    return o


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    dataset = args.dataset
    alloc_file = args.alloc_file
    local_search_partial_returned_order = args.local_search_partial_returned_order

    random.seed(args.seed)
    np.random.seed(args.seed)

    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)

    with open(local_search_partial_returned_order, "rb") as f:
        partial_order = pickle.load(f)

    partial_order_list = partial_order_to_list(partial_order)

    remaining_agents = set(range(len(covs))) - set(partial_order_list)

    best_revs = np.argsort(-1 * scores, axis=0)

    m, n = scores.shape

    # n_tries = 100
    # best_usw = -1
    # best_order = None
    # for _ in range(n_tries):
    #     total_order = deepcopy(partial_order_list)
    #     for a in remaining_agents:
    #         total_order.insert(random.randint(0, len(total_order)), a)
    #     usw_, _, _ = rr_usw(total_order, scores, covs, loads, best_revs)
    #     if usw_ > best_usw:
    #         print("New best usw: ", usw_/n)
    #         best_usw = usw_
    #         best_order = total_order


    total_order = deepcopy(partial_order_list)
    num_iters = len(remaining_agents)
    for _ in range(num_iters):
        print("adding another agent")
        best_usw = -1
        best_order = None
        best_agent = -1

        num_choices = 5

        num_inserts = 2
        # insertion_points = [math.floor(len(total_order)*i/num_inserts) for i in range(1, num_inserts)]
        insertion_points = random.sample(range(len(total_order)), num_inserts)

        # print(len(remaining_agents))
        samp = None
        if len(remaining_agents) > num_choices:
            samp = random.sample(remaining_agents, num_choices)
        else:
            samp = remaining_agents
        for a in samp:
            for ip in insertion_points:
                o = deepcopy(total_order)
                o.insert(ip, a)
                usw_, _, _ = rr_usw(total_order, scores, covs, loads, best_revs)
                if usw_ > best_usw:
                    print("New best usw: ", usw_/n)
                    best_usw = usw_
                    best_order = o
                    best_agent = a
        total_order = best_order
        remaining_agents = remaining_agents - {best_agent}
        print(len(set(total_order)))
        print(len(remaining_agents))
        print(n)
        # print("new total_order: ", total_order)

    alloc, _, _ = rr(total_order, scores, covs, loads, best_revs)

    save_alloc(alloc, alloc_file)
    print_stats(alloc, scores, covs)


    # with open("complete_order_cvpr_debug", "wb") as f:
    #     pickle.dump(complete_seln_order, f)
    # with open("partial_order_cvpr_debug", "wb") as f:
    #     pickle.dump(partial_seln_order, f)
    # save_alloc(complete_alloc, "complete_greedy_cvpr_debug")
    # save_alloc(partial_alloc, "partial_greedy_cvpr_debug")



