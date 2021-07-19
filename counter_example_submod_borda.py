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

    best_goods = np.argsort(-1 * val_fns, axis=1)

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


def fn_rr_usw(order_tuples, val_fns, fn=np.mean, num_samples=None, use_mask=False):
    n, m = val_fns.shape

    best_goods = np.argsort(-1 * val_fns, axis=1)
    limit = 1

    remaining_slots = set(range(n)) - set([t[1] for t in order_tuples])
    unassigned_agents = sorted(list(set(range(n)) - set([t[0] for t in order_tuples])))

    mask = np.ones(n)
    if use_mask:
        mask[unassigned_agents] = 0

    usws = []
    if not num_samples:
        for perm in permutations(remaining_slots):
            remaining = np.ones((m))
            matrix_alloc = np.zeros((n, m), dtype=np.bool)

            filled_out_order = deepcopy(order_tuples)
            filled_out_order |= set(zip(unassigned_agents, perm))
            order = sorted(filled_out_order, key=lambda x: x[1])
            ordering = [t[0] for t in order]

            # while np.sum(remaining):
            for _ in range(limit):
                for a in ordering:
                    for r in best_goods[a, :]:
                        if remaining[r]:
                            remaining[r] -= 1
                            matrix_alloc[a, r] = 1
                            break
            usws.append(np.sum(matrix_alloc * val_fns * mask.reshape(-1, 1)))
    print(order_tuples)
    print(usws)
    print(fn(usws))

    return fn(usws)


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
        graph.add_edge(a, 1, weight=0, capacity=rounds * (n - len(ordering)))

    return graph


def get_remaining_from_flow_result(flowDict, m, n):
    goods = list(range(m))
    goods = [i + 2 for i in goods]  # the "2" are the source and sink

    remaining = np.ones((m))

    for g in goods:
        for a in flowDict[g]:
            if flowDict[g][a] == 1:
                remaining[g - 2] -= 1

    return remaining


def preempted_rr_usw(ordering, val_fns, rounds=1):
    if not len(ordering):
        return 0

    matrix_alloc = np.zeros((val_fns.shape), dtype=np.bool)

    n, m = val_fns.shape

    # remaining = np.ones((m))

    best_goods = np.argsort(-1 * val_fns, axis=1)

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


# def check_submodularity(val_fns, fun, t):
#     n, m = val_fns.shape
#
#     # For all proper subsets of agents
#     for a_set in powerset(range(n)):
#         a_set = set(a_set)
#         if a_set != set(range(n)) and len(a_set):
#             for subset in powerset(a_set):
#                 if len(subset) > 2:
#                     subset = set(subset)
#                     if subset != a_set:
#                         for e in set(range(n)) - a_set:
#                             for order in permutations(a_set):
#                                 # print("order up top: ", order)
#                                 order = list(order)
#                                 for i in range(len(a_set)):
#                                     # print("calling usw calls")
#                                     suborder = [j for j in order if j in subset]
#                                     # print("order: ", order)
#                                     # print(a_set)
#                                     # print(set(range(n)))
#                                     # print("suborder: ", suborder)
#                                     # print(i)
#                                     # print(e)
#                                     usw_order = fun(order, val_fns)
#                                     usw_suborder = fun(suborder, val_fns)
#                                     order.insert(i, e)
#                                     suborder = [j for j in order if j in (subset | {e})]
#                                     usw_order_e = fun(order, val_fns)
#                                     usw_suborder_e = fun(suborder, val_fns)
#                                     order.remove(e)
#                                     suborder.remove(e)
#                                     sub1 = t(usw_order_e) - t(usw_order)
#                                     sub2 = t(usw_suborder_e) - t(usw_suborder)
#                                     if not np.isclose(sub1, sub2) and sub1 > sub2:
#                                         print(t(usw_order_e))
#                                         print(t(usw_suborder_e))
#                                         print(t(usw_order))
#                                         print(t(usw_suborder))
#                                         print(t(usw_order_e) - t(usw_order))
#                                         print(t(usw_suborder_e) - t(usw_suborder))
#                                         print(order, suborder, e, i)
#                                         return False
#     return True


def generate_X_Y_e(n):
    # For all proper subsets of agents
    for a_set in powerset(range(n)):
        a_set = sorted(list(a_set))
        if len(a_set) and len(a_set) < n:
            # Take all ways to assign these agents to positions in the ordering
            for positions in combinations(range(n), len(a_set)):
                for pos_perm in permutations(positions):
                    Y = list(zip(a_set, pos_perm))
                    for X in powerset(Y):
                        if len(X) < len(Y):
                            remaining_agents = set(range(n)) - set(a_set)
                            remaining_positions = set(range(n)) - set(positions)
                            for e in product(remaining_agents, remaining_positions):
                                yield set(X), set(Y), e


def greedy_completion(S_in, val_fns):
    S = deepcopy(S_in)
    n, m = val_fns.shape
    agents = [x[0] for x in S]
    positions = [x[1] for x in S]
    remaining_positions = sorted(list(set(range(n)) - set(positions)))
    remaining_agents = set(range(n)) - set(agents)

    for p in remaining_positions:
        # Add the agent which adds most value.
        best_usw = 0
        agent_to_add = None
        for a in remaining_agents:
            curr_usw = rr_usw([x[0] for x in sorted(list(S | {(a, p)}), key=lambda x: x[1])], val_fns, rounds=1)
            if curr_usw > best_usw:
                agent_to_add = a
                best_usw = curr_usw
        S.add((agent_to_add, p))

    return S


def best_greedy_completion(S_in, val_fns):
    S = deepcopy(S_in)
    n, m = val_fns.shape
    agents = [x[0] for x in S]
    positions = [x[1] for x in S]
    remaining_positions = set(range(n)) - set(positions)
    remaining_agents = set(range(n)) - set(agents)

    # Agents which were the greedy choice in some iteration
    tuples_to_add = []
    finished_first_iter = False

    best_completion = None
    global_best_usw = 0

    def convert_to_flat_order(set_of_tuples):
        return [x[0] for x in sorted(list(set_of_tuples), key=lambda x: x[1])]

    while len(tuples_to_add) or not finished_first_iter or len(S) < n:
        finished_first_iter = True
        # Add one of the agents which add most value.
        if len(S) != n:
            p = min(remaining_positions - {x[1] for x in S})

            best_usw = 0
            local_tuples_to_add = {}
            for a in remaining_agents - {x[0] for x in S}:
                curr_usw = rr_usw(convert_to_flat_order(S | {(a, p)}), val_fns, rounds=1)
                if curr_usw >= best_usw:
                    local_tuples_to_add[(a, p)] = curr_usw
                    best_usw = curr_usw
            local_tuples_to_add = [t[0] for t in local_tuples_to_add.items() if t[1] == best_usw]
            tuples_to_add.extend(local_tuples_to_add)
        else:
            local_best_usw = rr_usw(convert_to_flat_order(S), val_fns, rounds=1)
            if local_best_usw >= global_best_usw:
                global_best_usw = local_best_usw
                best_completion = deepcopy(S)

        next_addition = tuples_to_add.pop(-1)  # DFS will work best here.

        if next_addition[1] < len(S):
            S = {s for s in S if (s[1] < next_addition[1] or s[1] not in remaining_positions)}

        S.add(next_addition)

    # Potentially evaluate one last time
    local_best_usw = rr_usw(convert_to_flat_order(S), val_fns, rounds=1)
    if local_best_usw >= global_best_usw:
        global_best_usw = local_best_usw
        best_completion = deepcopy(S)

    return best_completion


def check_submodularity(val_fns):
    n, m = val_fns.shape

    # Loop over subsets of tuples which are valid, and compare against all valid supersets, what happens when
    # you add an element?
    func = np.max
    # func = lambda x: 1*x
    # def func(S):
    #     return S[0]

    for X, Y, e in generate_X_Y_e(n):
        if len(X):
        # usw_Y = fn_rr_usw(best_greedy_completion(Y, val_fns), val_fns, fn=func)
        # usw_X = fn_rr_usw(best_greedy_completion(X, val_fns), val_fns, fn=func)
            usw_Y = fn_rr_usw(Y, val_fns, fn=func, use_mask=False)
            usw_X = fn_rr_usw(X, val_fns, fn=func, use_mask=False)

            Y_e = deepcopy(Y)
            Y_e.add(e)
            X_e = deepcopy(X)
            X_e.add(e)

            # usw_Y_e = fn_rr_usw(best_greedy_completion(Y_e, val_fns), val_fns, fn=func)
            # usw_X_e = fn_rr_usw(best_greedy_completion(X_e, val_fns), val_fns, fn=func)
            usw_Y_e = fn_rr_usw(Y_e, val_fns, fn=func, use_mask=False)
            usw_X_e = fn_rr_usw(X_e, val_fns, fn=func, use_mask=False)

            # log_base = 2
            # addv_factor = .3 or .411
            # addv_factor = -3.5

            # _usw_Y = usw_Y * math.log(len(Y) + addv_factor)
            # _usw_X = usw_X * math.log(len(X) + addv_factor)
            # _usw_Y_e = usw_Y_e * math.log(len(Y) + 1 + addv_factor)
            # _usw_X_e = usw_X_e * math.log(len(X) + 1 + addv_factor)
            # usw_Y = math.log(usw_Y, log_base)
            # usw_X = math.log(usw_X, log_base)
            # usw_Y_e = math.log(usw_Y_e, log_base)
            # usw_X_e = math.log(usw_X_e, log_base)

            sub1 = usw_Y_e*np.log((len(Y)+1)) - usw_Y*np.log(len(Y))
            sub2 = usw_X_e*np.log((len(X)+1)) - usw_X*np.log(len(X))

            # if len(X) == 1 and (1, 1) in X and (0, 2) in Y and e == (2,0):
            #     print(X)
            #     print(Y)
            #     print(e)
            #     print()
            #
            #     print(usw_Y_e)
            #     print(usw_Y)
            #
            #     print(usw_X_e)
            #     print(usw_X)

            if not np.isclose(sub1, sub2) and sub1 > sub2:
                print(val_fns)

                print(Y)
                print(X)
                print(e)

                print(usw_Y_e)
                print(usw_Y)

                print(usw_X_e)
                print(usw_X)

                print(sub1)
                print(sub2)
                return False

    return True


def to_borda(val_fns):
    return np.argsort(np.argsort(val_fns, axis=1), axis=1) + 1


def harmonic_val_fns(n, m):
    vfs = []
    for i in range(n):
        vfs.append(sorted([1 / (v + 1) for v in range(m)], key=lambda x: random.random()))
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
        for _ in range(m - 1):
            vf.append(n * vf[-1] + 1)
        vfs.append(sorted([1 / v for v in vf], key=lambda x: random.random()))
    return np.array(vfs)


if __name__ == "__main__":
    # exhaustive_sim_triplet(efx_among_triplet)
    for s in range(100):
        if s % 1 == 0:
            print(s)
        s = 1
        np.random.seed(s + 1000)
        random.seed(s + 1000)

        n = 4
        m = 4
        valuations = np.random.rand(n, m)
        # print(valuations)
        # print(np.sum(valuations, axis=1))
        # print(np.sum(valuations, axis=1))
        valuations = to_borda(valuations)
        # valuations = valuations * 1/np.sum(valuations, axis=1).reshape((-1,1))

        # valuations = np.array([[3, 4, 1, 2],[2, 1, 3, 4], [1, 3, 2, 4], [2, 4, 1, 3]])
        # valuations = binary_val_fns(n, m)
        # while np.any(np.sum(valuations, axis=1) == 0):
        #     valuations = binary_val_fns(n, m)
        # valuations = harmonic_val_fns(n, m)
        # valuations = rapidly_decaying(n, m)

    # valuations = np.array([[3, 2, 1],
    #                        [2, 3, 1],
    #                        [1, 3, 2]])

    # for v1 in permutations(range(1, 4)):
    #     for v2 in permutations(range(1, 4)):
    #         for v3 in permutations(range(1, 4)):
    #             valuations = np.array([list(v1), list(v2), list(v3)])
    #             print(valuations)
        if not check_submodularity(val_fns=valuations):
            print(s)
            sys.exit(0)
