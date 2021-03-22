import math
import networkx as nx
import numpy as np
from copy import deepcopy
from itertools import permutations, chain, combinations, product
import random


def is_efx(alloc1, alloc2, a1, a2, val_fns):
    for i in alloc1:
        if np.sum(val_fns[a2, alloc2]) < np.sum(val_fns[a2, alloc1]) - val_fns[a2, i]:
            return False
    for i in alloc2:
        if np.sum(val_fns[a1, alloc1]) < np.sum(val_fns[a1, alloc2]) - val_fns[a1, i]:
            return False
    return True


def get_valuations(alloc1, alloc2, val_fns):
    return np.sum(val_fns[0, alloc1]), np.sum(val_fns[1, alloc2])


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def rr_usw(ordering, val_fns, rounds=1):
    if not len(ordering):
        return 0

    matrix_alloc = np.zeros((val_fns.shape), dtype=np.bool)

    n, m = val_fns.shape

    remaining = np.ones((m))

    best_goods = np.argsort(-1*val_fns, axis=1)

    # if not allocate_all:
    #     for _ in range(math.floor(m / n)):
    #         for a in ordering:
    #             for r in best_goods[a, :]:
    #                 if remaining[r]:
    #                     remaining[r] -= 1
    #                     matrix_alloc[a, r] = 1
    #                     break
    # else:
    for _ in range(rounds):
        for a in ordering:
            for r in best_goods[a, :]:
                if remaining[r]:
                    remaining[r] -= 1
                    matrix_alloc[a, r] = 1
                    break
# print("in usw")
    # print(matrix_alloc)
    # print(val_fns)
    return np.sum(matrix_alloc * val_fns)


# Find a min-cost flow which sends rounds * (n - len(ordering)) units of flow through the agents in ordering.
# We will pretend like these goods have been "taken" when we then compute the rr_usw on the agents in ordering.
def draw_flow_graph(ordering, m, n, scores, rounds):
    # print(ordering)
    # print(m)
    # print(n)
    graph = nx.DiGraph()

    goods = list(range(m))
    goods = [i + 2 for i in goods]  # the "2" are the source and sink
    agents = list(range(n))
    agents = [i + len(goods) + 2 for i in agents]

    # supply_and_demand = rounds * min(len(ordering), (n - len(ordering)))
    supply_and_demand = rounds * (n - len(ordering))
    # print(supply_and_demand)
    graph.add_node(0, demand=int(-1 * supply_and_demand))
    graph.add_node(1, demand=int(supply_and_demand))

    for g in goods:
        graph.add_node(g, demand=0)
    for a in agents:
        graph.add_node(a, demand=0)

    # Draw edges from goods to agents ~in the ordering~
    W = int(1e10)
    for a in ordering:
        for g in goods:
            graph.add_edge(g, a + len(goods) + 2, weight=-1 * int(W * scores[a, g - 2]),
                           capacity=1)
    # Draw edges from source to goods
    for g in goods:
        graph.add_edge(0, g, weight=0, capacity=1)
    # Draw edges from papers to sink
    for a in agents:
        graph.add_edge(a, 1, weight=0, capacity=rounds*(n-len(ordering)))

    return graph


def get_remaining_from_flow_result(flowDict, m, n):
    goods = list(range(m))
    goods = [i + 2 for i in goods]  # the "2" are the source and sink

    remaining = np.ones((m))

    for g in goods:
        for a in flowDict[g]:
            if flowDict[g][a] == 1:
                remaining[g-2] -= 1

    return remaining


def preempted_rr_usw(ordering, val_fns, rounds=1):
    if not len(ordering):
        return 0

    matrix_alloc = np.zeros((val_fns.shape), dtype=np.bool)

    n, m = val_fns.shape

    # remaining = np.ones((m))

    best_goods = np.argsort(-1*val_fns, axis=1)

    # construct a network flow which will determine which goods get "pre-emtped"
    graph = draw_flow_graph(ordering, m, n, val_fns, rounds)
    # solve it
    flow_dict = nx.min_cost_flow(graph)
    remaining = get_remaining_from_flow_result(flow_dict, m, n)

    # if not allocate_all:
    #     for _ in range(math.floor(m / n)):
    #         for a in ordering:
    #             for r in best_goods[a, :]:
    #                 if remaining[r]:
    #                     remaining[r] -= 1
    #                     matrix_alloc[a, r] = 1
    #                     break
    # else:
    for _ in range(rounds):
        for a in ordering:
            for r in best_goods[a, :]:
                if remaining[r]:
                    remaining[r] -= 1
                    matrix_alloc[a, r] = 1
                    break
# print("in usw")
    # print(matrix_alloc)
    # print(val_fns)
    return np.sum(matrix_alloc * val_fns)


def check_submodularity(val_fns, fun, t):
    n, m = val_fns.shape

    # For all proper subsets of agents
    for a_set in powerset(range(n)):
        a_set = set(a_set)
        if a_set != set(range(n)) and len(a_set):
            for subset in powerset(a_set):
                if len(subset) > 2:
                    subset = set(subset)
                    if subset != a_set:
                        for e in set(range(n)) - a_set:
                            for order in permutations(a_set):
                                # print("order up top: ", order)
                                order = list(order)
                                for i in range(len(a_set)):
                                    # print("calling usw calls")
                                    suborder = [j for j in order if j in subset]
                                    # print("order: ", order)
                                    # print(a_set)
                                    # print(set(range(n)))
                                    # print("suborder: ", suborder)
                                    # print(i)
                                    # print(e)
                                    usw_order = fun(order, val_fns)
                                    usw_suborder = fun(suborder, val_fns)
                                    order.insert(i, e)
                                    suborder = [j for j in order if j in (subset | {e})]
                                    usw_order_e = fun(order, val_fns)
                                    usw_suborder_e = fun(suborder, val_fns)
                                    order.remove(e)
                                    suborder.remove(e)
                                    sub1 = t(usw_order_e) - t(usw_order)
                                    sub2 = t(usw_suborder_e) - t(usw_suborder)
                                    if not np.isclose(sub1, sub2) and sub1 > sub2:
                                        print(t(usw_order_e))
                                        print(t(usw_suborder_e))
                                        print(t(usw_order))
                                        print(t(usw_suborder))
                                        print(t(usw_order_e) - t(usw_order))
                                        print(t(usw_suborder_e) - t(usw_suborder))
                                        print(order, suborder, e, i)
                                        return False
    return True


def to_borda(val_fns):
    return np.argsort(np.argsort(val_fns, axis=1), axis=1) + 1


def harmonic_val_fns(n, m):
    vfs = []
    for i in range(n):
        vfs.append(sorted([1/(v+1) for v in range(m)], key=lambda x: random.random()))
    return np.array(vfs)


def binary_val_fns(n, m):
    vfs = []
    for i in range(n):
        vf = []
        for j in range(m):
            vf.append(random.randint(0, 1))
        vfs.append(vf)
    return np.array(vfs)


def rapidly_decaying(n, m):
    vfs = []
    for i in range(n):
        vf = [1]
        for _ in range(m-1):
            vf.append(n*vf[-1]+1)
        vfs.append(sorted([1 / v for v in vf], key=lambda x: random.random()))
    return np.array(vfs)


if __name__ == "__main__":
    # exhaustive_sim_triplet(efx_among_triplet)
    for s in range(100000):
        if s % 50 == 0:
            print(s)
        np.random.seed(s)
        random.seed(s)

        n = 5
        m = 5
        valuations = np.random.rand(n, m)
        # print(valuations)
        # print(np.sum(valuations, axis=1))
        # valuations = valuations * 1/np.sum(valuations, axis=1).reshape((-1,1))
        # print(np.sum(valuations, axis=1))
        valuations = to_borda(valuations)
        # valuations = np.array([[3, 4, 1, 2],[2, 1, 3, 4], [1, 3, 2, 4], [2, 4, 1, 3]])
        # valuations = binary_val_fns(n, m)
        # valuations = harmonic_val_fns(n, m)
        # valuations = rapidly_decaying(n, m)

        transformation = lambda x: x - 25*np.log(x) if x else 0
        # transformation = lambda x: 1*x

        print(valuations)
        if not np.any(np.sum(valuations, axis=1) == 0) and not check_submodularity(valuations, preempted_rr_usw, transformation):
            break
            # found_order = False
            # for order in permutations(range(m)):
            #     if check_submodularity(valuations[:, order], rr_usw):
            #         found_order = True
            #         break
            # if not found_order:
            #     break
