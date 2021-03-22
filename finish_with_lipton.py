# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import os
import sys

from autoassigner import *
from copy import deepcopy
import networkx as nx

from greedy_rr import rr
from utils import *


def remove_cycles(alloc, eg):
    while True:
        try:
            cyc = nx.find_cycle(eg)
            tmp = alloc[cyc[0][0]]
            for e in cyc[-1]:
                # Make the swap
                alloc[e[0]] = alloc[e[1]]
                # Remove edge
                eg.remove_edge(e[0], e[1])

            alloc[cyc[-1][0]] = tmp
            eg.remove_edge(cyc[-1][0], cyc[-1][1])

            # Reroute edges... note this includes edges such as 0 -> 2 becoming 0 -> 1.
            new_dests = {e[1]: e[0] for e in cyc}
            for e in eg.edges():
                if e[1] in new_dests:
                    eg.remove_edge(e[0], e[1])
                    eg.add_edge(e[0], new_dests[e[1]])
        except nx.NetworkXNoCycle:
            return alloc, eg


def create_envy_graph(alloc, scores):
    envy_g = nx.DiGraph()
    envy_g.add_nodes_from(alloc.keys())
    for paper1 in alloc:
        for paper2 in alloc:
            if paper1 != paper2:
                other = get_valuation(paper1, alloc[paper2], scores)
                curr = get_valuation(paper1, alloc[paper1], scores)
                if other > curr:
                    envy_g.add_edge(paper1, paper2)
    return envy_g


def matrix_from_alloc(alloc, loads, scores):
    matrix_alloc = np.zeros(scores.shape)
    for a, revs in alloc:
        for r in revs:
            matrix_alloc[r, a] = 1
            loads[r] -= 1
    return matrix_alloc, loads


def add_reviewer(alloc, scores, loads, covs, eg):
    reversed = eg.reverse(copy=False)
    selected_agent = None
    best_value = -1
    for a in alloc:
        if len(alloc[a]) < covs[a] and not reversed.adj[a]:
            top_score = np.max(scores[:, a] * (loads > 0))
            if top_score > best_value:
                selected_agent = a
                best_value = top_score

    r = np.argmax(scores[:, selected_agent] * (loads > 0))[0]
    alloc[selected_agent].append(r)
    loads[r] -= 1
    print("adding %d to %d" % (r, selected_agent))

    # Add any new envy edges, if applicable
    for a in alloc:
        if a != selected_agent:
            other = get_valuation(a, alloc[selected_agent], scores)
            curr = get_valuation(a, alloc[a], scores)
            if other > curr:
                eg.add_edge(a, selected_agent)

    return alloc, loads, eg


def finish_with_lipton(partial_alloc, scores, covs, loads):
    # Have a function which creates the envy graph, finds and removes cycles.
    # Then find all nodes with no in-edges and less than c items.
    # Then add to one of those.
    # Hopefully we can do this until the end.

    alloc = partial_alloc
    matrix_alloc, loads = matrix_from_alloc(alloc, loads, scores)
    eg = create_envy_graph(alloc, scores)

    while np.any(np.sum(matrix_alloc, axis=0) < covs):
        alloc, eg = remove_cycles(alloc, eg)

        # Add a reviewer to a paper.
        alloc, eg, loads = add_reviewer(alloc, scores, loads, covs, eg)

        matrix_alloc, loads = matrix_from_alloc(alloc, loads, scores)

    return alloc


def partial_order_to_alloc(order, scores, covs, loads):
    best_revs = np.argsort(-1 * scores, axis=0)
    o = sorted(order, key=lambda x: x[1])
    o = [t[0] for t in o]
    return rr(o, scores, covs, loads, best_revs)[0]


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    dataset = args.dataset
    alloc_file = args.alloc_file
    local_search_partial_returned_order = args.local_search_partial_returned_order

    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)

    with open(local_search_partial_returned_order, "rb") as f:
        partial_order = pickle.load(f)

    partial_alloc = partial_order_to_alloc(partial_order, scores, covs, loads)

    alloc = finish_with_lipton(partial_alloc, scores, covs, loads)

    save_alloc(alloc, alloc_file)
    print_stats(alloc, scores, covs)


    # with open("complete_order_cvpr_debug", "wb") as f:
    #     pickle.dump(complete_seln_order, f)
    # with open("partial_order_cvpr_debug", "wb") as f:
    #     pickle.dump(partial_seln_order, f)
    # save_alloc(complete_alloc, "complete_greedy_cvpr_debug")
    # save_alloc(partial_alloc, "partial_greedy_cvpr_debug")



