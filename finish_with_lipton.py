# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import os
import sys

from autoassigner import *
from copy import deepcopy
import networkx as nx

from greedy_rr import rr
from utils import *


def remove_cycles(matrix_alloc, eg):
    # while True:
    try:
        cyc = nx.find_cycle(eg)
        print(cyc)
        print(" ".join([str(np.where(matrix_alloc[:, e[0]])[0]) for e in cyc]))

        tmp = np.copy(matrix_alloc[:, cyc[0][0]])
        for e in cyc[:-1]:
            # Make the swap
            # print(e)
            matrix_alloc[:, e[0]] = np.copy(matrix_alloc[:, e[1]])
            # Remove edge
            eg.remove_edge(e[0], e[1])

        matrix_alloc[:, cyc[-1][0]] = tmp
        eg.remove_edge(cyc[-1][0], cyc[-1][1])

        # Reroute edges... note this includes edges such as 0 -> 2 becoming 0 -> 1.
        revd = eg.reverse()
        for e in cyc:
            for envious in revd.adj[e[1]]:
                eg.remove_edge(envious, e[1])
                eg.add_edge(envious, e[0])

        print(" ".join([str(np.where(matrix_alloc[:, e[0]])[0]) for e in cyc]))
        return matrix_alloc, eg

    except nx.NetworkXNoCycle:
        return matrix_alloc, eg


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


def matrix_from_alloc(alloc, scores):
    matrix_alloc = np.zeros(scores.shape)
    for a, revs in alloc.items():
        for r in revs:
            matrix_alloc[r, a] = 1
    return matrix_alloc


def add_reviewer(matrix_alloc, scores, loads, covs, eg):
    revd = eg.reverse(copy=False)
    selected_agent = None
    best_value = -1

    for a in range(matrix_alloc.shape[0]):
        if np.sum(matrix_alloc[:, a]) < covs[a]:
            if not revd.adj[a]:
                top_score = np.max(scores[:, a] * (loads > 0) * (1 - matrix_alloc[:, a]))
                if top_score > best_value:
                    selected_agent = a
                    best_value = top_score

    if selected_agent:
        r = np.argmax((scores[:, selected_agent]+.01) * (loads > 0) * (1 - matrix_alloc[:, selected_agent])).item()
        matrix_alloc[r, selected_agent] = 1
        loads[r] -= 1

        print("adding %d to %d" % (r, selected_agent))

        # Add any new envy edges, if applicable
        for a in range(matrix_alloc.shape[0]):
            if a != selected_agent:
                other = np.sum(matrix_alloc[:, selected_agent] * scores[:, a])
                curr = np.sum(matrix_alloc[:, a] * scores[:, a])
                if other > curr:
                    eg.add_edge(a, selected_agent)
    else:
        # print(np.where(np.sum(matrix_alloc, axis=0) < covs)[0])
        # alloc = {}
        # for a in range(matrix_alloc.shape[1]):
        #     alloc[a] = np.where(matrix_alloc[:, a])[0].tolist()
        print("no selected agent")
        for a in range(matrix_alloc.shape[0]):
            if np.sum(matrix_alloc[:, a]) < covs[a]:
                print(a)
                print(np.sum(matrix_alloc[:, a]))
                print(revd.adj[a])
                print(np.sum(loads > 0))
                print(np.sum((1-matrix_alloc[:, a])))
                print(np.sum((loads > 0) * (1-matrix_alloc[:, a])))
        # print(sorted(alloc.items()))
        # print(revd.adj[selected_agent])
        # print()
        # print()

    return matrix_alloc, loads, eg, selected_agent


def finish_with_lipton(partial_alloc, scores, covs, loads):
    # Have a function which creates the envy graph, finds and removes cycles.
    # Then find all nodes with no in-edges and less than c items.
    # Then add to one of those.
    # Hopefully we can do this until the end.

    alloc = partial_alloc
    matrix_alloc = matrix_from_alloc(alloc, scores)
    eg = create_envy_graph(alloc, scores)

    while np.any(np.sum(matrix_alloc, axis=0) < covs):
        # Add a reviewer to a paper.
        matrix_alloc, loads, eg, success = add_reviewer(matrix_alloc, scores, loads, covs, eg)
        if not success:
            matrix_alloc, eg = remove_cycles(matrix_alloc, eg)
            matrix_alloc, loads, eg, _ = add_reviewer(matrix_alloc, scores, loads, covs, eg)

    alloc = {}
    for a in range(matrix_alloc.shape[1]):
        alloc[a] = np.where(matrix_alloc[:, a])[0].tolist()

    return alloc


def partial_order_to_alloc(order, scores, covs, loads):
    best_revs = np.argsort(-1 * scores, axis=0)
    o = sorted(order, key=lambda x: x[1])
    o = [t[0] for t in o]
    return rr(o, scores, covs, loads, best_revs)


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

    print(np.min(loads))
    partial_alloc, loads, _ = partial_order_to_alloc(partial_order, scores, covs, loads)
    print(np.min(loads))
    for a in set(range(len(covs))) - set(partial_alloc.keys()):
        partial_alloc[a] = []

    print(partial_alloc)

    alloc = finish_with_lipton(partial_alloc, scores, covs, loads)

    save_alloc(alloc, alloc_file)
    print_stats(alloc, scores, covs)


    # with open("complete_order_cvpr_debug", "wb") as f:
    #     pickle.dump(complete_seln_order, f)
    # with open("partial_order_cvpr_debug", "wb") as f:
    #     pickle.dump(partial_seln_order, f)
    # save_alloc(complete_alloc, "complete_greedy_cvpr_debug")
    # save_alloc(partial_alloc, "partial_greedy_cvpr_debug")



