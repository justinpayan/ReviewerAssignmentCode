import math
import networkx as nx
import random
import sys

from copy import deepcopy
from eit_by_agents import *
from itertools import product


def get_all_partitions(g_set, n, orig_n):
    # if n == orig_n:
    #     print("STARTING")
    # print(g_set, n, orig_n)
    if n == 1:
        # print("Yielding")
        # print({tuple(g_set)})
        yield {tuple(g_set)}
    else:
        for s in powerset(g_set):
            # print(s)
            # print(s)
            # print(g_set)
            remaining_goods = g_set - set(s)
            # print(remaining_goods)
            for sub_alloc in get_all_partitions(remaining_goods, n - 1, orig_n):
                sa_copy = deepcopy(sub_alloc)
                sa_copy.add(s)
                # print(sub_alloc)
                yield sa_copy
        # print("next\n")


def get_all_equal_partitions(g_set, n, orig_n, lower_threshold, upper_threshold):
    if n == 1:
        yield {tuple(g_set)}
    else:
        for s in powerset(g_set):
            if lower_threshold <= len(s) <= upper_threshold:
                remaining_goods = g_set - set(s)
                for sub_alloc in get_all_equal_partitions(remaining_goods, n - 1, orig_n,
                                                          lower_threshold, upper_threshold):
                    sa_copy = deepcopy(sub_alloc)
                    sa_copy.add(s)
                    yield sa_copy


def get_all_allocs_with_heuristic(agents, goods):
    g_set = set(goods)
    # print(len(agents))
    lt = math.floor(len(goods) / len(agents))
    ut = math.ceil(len(goods) / len(agents))
    for partition in get_all_equal_partitions(g_set, len(agents), len(agents), lt, ut):
        # print(partition)
        part_list = list(partition)
        # Condition may be useful for speeding up search
        if min([len(x) for x in part_list]) >= math.floor(len(goods) / len(agents)):
            if len(part_list) <= len(agents):
                while len(part_list) < len(agents):
                    part_list.append(())
                for alloc in permutations(part_list):
                    yield alloc


def get_all_allocs(agents, goods):
    g_set = set(goods)
    # print(len(agents))
    for partition in get_all_partitions(g_set, len(agents), len(agents)):
        # print(partition)
        part_list = list(partition)
        # Condition may be useful for speeding up search
        # if min([len(x) for x in part_list]) >= 4:
        if len(part_list) <= len(agents):
            while len(part_list) < len(agents):
                part_list.append(())
            for alloc in permutations(part_list):
                yield alloc


def get_splits(all_goods):
    for s in powerset(all_goods):
        yield s, list(set(all_goods) - set(s))


def get_value(agent, goods, val_fns):
    v = val_fns[agent]
    return sum(v[g] for g in goods)


def pareto_dominates(all1, all2, val_fns):
    one_improvement = False
    for agent in range(len(all1)):
        if get_value(agent, all1[agent], val_fns) > get_value(agent, all2[agent], val_fns):
            one_improvement = True
        elif get_value(agent, all1[agent], val_fns) < get_value(agent, all2[agent], val_fns):
            return False
    return one_improvement


def po(alloc, val_fns):
    # for other_alloc in get_all_allocs(list(range(len(alloc))), list(range(len(val_fns[0])))):
    for other_alloc in get_all_allocs(list(range(len(alloc))), list(val_fns[0])):
        if pareto_dominates(other_alloc, alloc, val_fns):
            return False
    return True


# Can we drop any item and have the result bundle not be envied by the other agent?
def efx(a1, a2, alloc, val_fns):
    a1_value = get_value(a1, alloc[a1], val_fns)
    a2_value = get_value(a2, alloc[a2], val_fns)

    a1_set = set(alloc[a1])
    a2_set = set(alloc[a2])

    for g in a1_set:
        if get_value(a2, a1_set - {g}, val_fns) > a2_value:
            return False

    for g in a2_set:
        if get_value(a1, a2_set - {g}, val_fns) > a1_value:
            return False

    return True


# pairwise efx means that it is envy free up to any item if you look at all edges in the graph
# pairwise efx on a clique is the same as efx.
def alloc_is_pefx(alloc, G, val_fns):
    for e in G.edges():
        if not efx(e[0], e[1], alloc, val_fns):
            return False
    return True


def compute_po_efx(G, val_fns):
    agents = list(val_fns.keys())
    goods = list(range(len(val_fns[0].values())))
    for alloc in get_all_allocs_with_heuristic(agents, goods):
        # print(alloc)
        if alloc_is_pefx(alloc, G, val_fns) and po(alloc, val_fns):
            # print("SUCCESS")
            return alloc
        # print("NO")
    print("FAILURE IN HEURISTIC SEARCH")
    for alloc in get_all_allocs(agents, goods):
        # print(alloc)
        if alloc_is_pefx(alloc, G, val_fns) and po(alloc, val_fns):
            # print("SUCCESS")
            return alloc
    print("FAIL")

    print("\n***********\n***********\n***********\n")

    return None


def find_po_and_efx_alloc_cgm20():
    agents = [0, 1, 2]
    G = nx.Graph()
    G.add_nodes_from(agents)
    G.add_edge(agents[0], agents[1])
    G.add_edge(agents[1], agents[2])
    G.add_edge(agents[2], agents[0])

    # Later, we should add epsilons
    val_fns = {0: {0: 8, 1: 2, 2: 12, 3: 2, 4: 0, 5: 17, 6: 1},
               1: {0: 5, 1: 0, 2: 9, 3: 4, 4: 10, 5: 0, 6: 3},
               2: {0: 0, 1: 0, 2: 0, 3: 0, 4: 9, 5: 10, 6: 2}}

    return compute_po_efx(G, val_fns)


# Search over all valuation functions in a specified range (maybe with some epsilons applied?)
# and see if we can find an instance in which there is no pefx and po allocation.
def run_exhaustive_sim(goods, algorithm):
    agents = [0, 1, 2]
    # val_options = list(range(1,7))
    val_options = [random.randint(1, 35) for _ in range(2)]
    val_options.append(1000)
    # val_options.extend([(i + (random.random() - .5)*.0001) for i in val_options])

    G = nx.Graph()
    G.add_nodes_from(agents)
    G.add_edge(agents[0], agents[1])
    G.add_edge(agents[1], agents[2])

    val_fns = {0: {}, 1: {}, 2: {}}

    r = [val_options] * (3 * len(goods))
    for v in product(*r):
        old_a0 = deepcopy(val_fns[0])

        i = 0
        for a in agents:
            val_fns[a] = {}
            for g in goods:
                val_fns[a][g] = v[i]
                i += 1

        if val_fns[0] != old_a0:
            print(val_fns)

        if random.random() < .001:
            alloc = algorithm(G, val_fns)
            if not alloc:
                print("total failure")
                print(val_fns)
                print(alloc)
                return 0
            # elif not po(alloc, val_fns):
            #     print("\n\n\nnot PO\n\n\n")
            #     print(val_fns)
            #     print(alloc)
                # return 0

    return 1


def rand_sim(goods, algorithm):
    agents = [0, 1, 2]

    G = nx.Graph()
    G.add_nodes_from(agents)
    G.add_edge(agents[0], agents[1])
    G.add_edge(agents[1], agents[2])

    val_fns = {0: {}, 1: {}, 2: {}}

    i = 0
    for a in agents:
        for g in goods:
            val_fns[a][g] = random.randint(1, 20)
            i += 1

    alloc = algorithm(G, val_fns)
    if not alloc:
        if not yairs_algo_reverse_order(G, val_fns):
            print("total failure")
            print(val_fns)
            # print(alloc)
            return 0
    # elif not po(alloc, val_fns):
    #     print("\n\n\nnot PO\n\n\n")
    #     print(alloc)
    #     return 0
    # print(alloc)

    return 1

# Does this algorithm return a PO and P-EFX allocation for this problem instance?
def run_sim(val_fns, algorithm):
    agents = [0, 1, 2]

    G = nx.Graph()
    G.add_nodes_from(agents)
    G.add_edge(agents[0], agents[1])
    G.add_edge(agents[1], agents[2])

    alloc = algorithm(G, val_fns)
    if not alloc:
        print("total failure")
        print(alloc)
        return 0
    # elif not po(alloc, val_fns):
    #     print("\n\n\nnot PO\n\n\n")
    #     print(alloc)
    #     return 0
    print(alloc)

    return 1


# I guess check all permutations?
def leximin(a1, a2, alloc, val_fns):
    a1_value = get_value(a1, alloc[a1], val_fns)
    a2_value = get_value(a2, alloc[a2], val_fns)

    a1_set = set(alloc[a1])
    a2_set = set(alloc[a2])

    a1_union_a2 = a1_set | a2_set

    total_a1 = get_value(a1, a1_union_a2, val_fns)
    total_a2 = get_value(a2, a1_union_a2, val_fns)

    if total_a1 == 0 or total_a2 == 0:
        return True

    a1_value /= total_a1
    a2_value /= total_a2

    for s in powerset(a1_union_a2):
        s = set(s)
        a1_s = get_value(a1, s, val_fns)/total_a1
        a2_rest = get_value(a2, a1_union_a2 - s, val_fns)/total_a2
        a2_s = get_value(a2, s, val_fns)/total_a2
        a1_rest = get_value(a1, a1_union_a2 - s, val_fns)/total_a1
        if a1_s >= a1_value and a2_rest >= a2_value and a1_s + a2_rest > a1_value + a2_value:
            return False
        elif a2_s >= a2_value and a1_rest >= a1_value and a1_s + a2_rest > a1_value + a2_value:
            return False

    return True


def is_pairwise_leximin(alloc, G, val_fns):
    for e in G.edges():
        if not leximin(e[0], e[1], alloc, val_fns):
            return False
    return True


def yairs_algo(G, val_fns):
    goods = list(range(len(val_fns[0].values())))
    # Compare the value on agent at start, then start + direction, etc.
    # Return true if ss1 is greater than ss2 at some idx in this manner.
    def subset_cmp(ss1, ss2, direction, start, v):
        if ss1 is None or ss2 is None:
            return True
        idx = start
        while idx <= 2 and idx >= 0:
            v1 = get_value(idx, ss1, v)
            v2 = get_value(idx, ss2, v)
            if v1 != v2:
                return v1 > v2
            idx += direction
        return True

    # Compare the value on agent at start, then start + direction, etc.
    # Return true if ss1 is less than ss2 at start or greater than ss2 after start.
    def subset_cmp_lowest_give_away(ss1, ss2, direction, start, v):
        if ss1 is None or ss2 is None:
            return True
        idx = start
        # while idx <= 2 and idx >= 0:
        #     v1 = get_value(idx, ss1, v)
        #     v2 = get_value(idx, ss2, v)
        #     if v1 != v2 and idx == start:
        #         return v1 < v2
        #     elif v1 != v2:
        #         return v1 > v2
        #     idx += direction
        # return True
        v1 = get_value(idx, ss1, v)
        v2 = get_value(idx, ss2, v)
        return v1 <= v2

    # Transfer the subset from a1 to a2 in allocation _alloc
    def flow_subset(_alloc, a1, a2, subset):
        flowed_alloc = deepcopy(_alloc)
        for g in subset:
            flowed_alloc[a1].remove(g)
            flowed_alloc[a2].append(g)
        return flowed_alloc

    # Try to transfer some subset of goods from a1 to a2 to make them pairwise EFX.
    # _fixed_good must stay with a1.
    def flow(_alloc, a1, a2, _fixed_good):
        if _alloc is None:
            return None

        pair_G = nx.Graph()
        pair_G.add_nodes_from([a1, a2])
        pair_G.add_edge(a1, a2)

        if alloc_is_pefx(_alloc, pair_G, val_fns):
            return _alloc

        chosen_subset = None
        for subset in powerset(_alloc[a1]):
            if _fixed_good not in subset:
                # if subset_cmp(subset, chosen_subset, a2-a1, a2, val_fns):
                if subset_cmp_lowest_give_away(subset, chosen_subset, a2 - a1, a1, val_fns):
                    flowed_alloc = flow_subset(_alloc, a1, a2, subset)

                    # if alloc_is_mms(flowed_alloc, pair_G, val_fns):
                    if alloc_is_pefx(flowed_alloc, pair_G, val_fns):
                        chosen_subset = deepcopy(subset)

        if chosen_subset is not None:
            flowed_alloc = flow_subset(_alloc, a1, a2, chosen_subset)
            return flowed_alloc
        else:
            return None

    def assign_left(alloc, G, val_fns, good, init_alloc):
        _alloc = deepcopy(alloc)
        _alloc[0].append(good)
        # Flow right
        while not alloc_is_pefx(_alloc, G, val_fns):
            prev_alloc = deepcopy(_alloc)
            # _alloc = flow(_alloc, 0, 1, good)
            _alloc = flow(_alloc, 0, 1, None)

            if _alloc is None:
                _alloc = init_alloc
                break
            if not alloc_is_pefx(_alloc, G, val_fns):
                # _alloc = flow(_alloc, 1, 2, good)
                _alloc = flow(_alloc, 1, 2, None)

            if prev_alloc == _alloc or _alloc is None:
                _alloc = init_alloc
                break
        return _alloc

    def assign_right(alloc, G, val_fns, good, init_alloc):
        _alloc = deepcopy(alloc)
        _alloc[2].append(good)
        # Flow right
        while not alloc_is_pefx(_alloc, G, val_fns):
            prev_alloc = deepcopy(_alloc)
            # _alloc = flow(_alloc, 2, 1, good)
            _alloc = flow(_alloc, 2, 1, None)

            if _alloc is None:
                _alloc = init_alloc
                break
            if not alloc_is_pefx(_alloc, G, val_fns):
                # _alloc = flow(_alloc, 1, 0, good)
                _alloc = flow(_alloc, 1, 0, None)

            if prev_alloc == _alloc or _alloc is None:
                _alloc = init_alloc
                break
        return _alloc

    def assign_center(alloc, G, val_fns, good, init_alloc):
        _alloc = deepcopy(alloc)
        _alloc[1].append(good)

        while not alloc_is_pefx(_alloc, G, val_fns):
            preflow = deepcopy(_alloc)
            _alloc = flow(_alloc, 1, 0, None)

            if _alloc is None:
                _alloc = init_alloc
                break
            if not alloc_is_pefx(_alloc, G, val_fns):
                # _alloc = flow(_alloc, 1, 0, good)
                _alloc = flow(_alloc, 1, 2, None)

            if preflow == _alloc or _alloc is None:
                _alloc = init_alloc
                break

            # if _alloc is None:
            #     _alloc = preflow
            #     _alloc = [_alloc[1], _alloc[0], _alloc[2]]

            # preflow = deepcopy(_alloc)
            # _alloc = flow(_alloc, 1, 2, good)
            # _alloc = flow(_alloc, 1, 2, None)

            # if _alloc is None:
            #     _alloc = preflow
            #     _alloc = [_alloc[0], _alloc[2], _alloc[1]]

            # if not alloc_is_pefx(_alloc, G, val_fns):
            #     print("Algorithm doesn't work")
            #     return None

        # if not alloc_is_pefx(_alloc, G, val_fns):
        #     if alloc_is_pefx([_alloc[0], _alloc[2], _alloc[1]], G, val_fns) and \
        #             get_value(1, _alloc[2], val_fns) >= get_value(1, _alloc[1], val_fns) and \
        #             get_value(2, _alloc[1], val_fns) >= get_value(2, _alloc[2], val_fns):
        #         _alloc = [_alloc[0], _alloc[2], _alloc[1]]
        #     elif alloc_is_pefx([_alloc[1], _alloc[0], _alloc[2]], G, val_fns) and \
        #             get_value(1, _alloc[0], val_fns) >= get_value(1, _alloc[1], val_fns) and \
        #             get_value(0, _alloc[1], val_fns) >= get_value(0, _alloc[0], val_fns):
        #         _alloc = [_alloc[1], _alloc[0], _alloc[2]]
        #     else:
        #         preflow = deepcopy(_alloc)
        #         # _alloc = flow(_alloc, 1, 0, good)
        #         _alloc = flow(_alloc, 1, 0, None)
        #
        #         if _alloc is None:
        #             _alloc = preflow
        #             _alloc = [_alloc[1], _alloc[0], _alloc[2]]
        #
        #         preflow = deepcopy(_alloc)
        #         # _alloc = flow(_alloc, 1, 2, good)
        #         _alloc = flow(_alloc, 1, 2, None)
        #
        #         if _alloc is None:
        #             _alloc = preflow
        #             _alloc = [_alloc[0], _alloc[2], _alloc[1]]
        #
        #         if not alloc_is_pefx(_alloc, G, val_fns):
        #             print("Algorithm doesn't work")
        #             return None

        return _alloc

    def leximin_value(alloc, val_fns):
        return sorted([get_value(i, alloc[i], val_fns) for i in range(3)])


    def restricted_po(alloc, val_fns):
        valid_items = set(alloc[0]) | set(alloc[1]) | set(alloc[2])
        new_val_fns = {0: {}, 1: {}, 2: {}}
        for a in val_fns:
            for g in valid_items:
                new_val_fns[a][g] = val_fns[a][g]
        return po(alloc, new_val_fns)

    # create allocation
    alloc = [[], [], []]
    # map from goods to order fn
    good_to_order_fn = {}
    for g in goods:
        # good_to_order_fn[g] = sorted([val_fns[i][g] for i in range(3)], reverse=True)
        good_to_order_fn[g] = sorted([val_fns[i][g] for i in range(3)])
    goods_ordered = sorted(goods, key=lambda x: good_to_order_fn[x])
    # goods_ordered = sorted(goods, key=lambda x: random.random())

    # Try agents in the order that they value the good
    # assign the good, check for pefx after flowing. If not, move to the next agent in the order
    for good in goods_ordered:
        init_alloc = deepcopy(alloc)

        a0 = assign_left(alloc, G, val_fns, good, init_alloc)
        a1 = assign_center(alloc, G, val_fns, good, init_alloc)
        a2 = assign_right(alloc, G, val_fns, good, init_alloc)

        # if a0 == init_alloc:
        #     print(alloc)
        #     print(val_fns)
        #     print(a2)
        #     print("\n\n\n")

        allocs = sorted([a0, a1, a2], key=lambda x: leximin_value(x, val_fns), reverse=True)
        # allocs = sorted([a0, a2], key=lambda x: leximax_value(x, val_fns), reverse=True)

        for a in allocs:
            # if a and a != init_alloc and alloc_is_pefx(a, G, val_fns) and restricted_po(a, val_fns):
            if a and a != init_alloc and alloc_is_pefx(a, G, val_fns):
                alloc = a
                break

        if alloc == init_alloc:
            print(good)
            print(alloc)
            return None
        # agent_order = np.argsort([val_fns[i][good] for i in range(3)])[::-1]
        # for a in agent_order:
        #     if a == 0:
        #         alloc = assign_left(alloc, G, val_fns, good, init_alloc)
        #     elif a == 2:
        #         alloc = assign_right(alloc, G, val_fns, good, init_alloc)
        #     elif a == 1:
        #         alloc = assign_center(alloc, G, val_fns, good, init_alloc)
        #
        #     if alloc != init_alloc:
        #         # We successfully assigned the good and got a p-efx allocation, move on
        #         break

    if not alloc_is_pefx(alloc, G, val_fns):
        return None
    else:
        return alloc



def yairs_algo_reverse_order(G, val_fns):
    goods = list(range(len(val_fns[0].values())))
    # Compare the value on agent at start, then start + direction, etc.
    # Return true if ss1 is greater than ss2 at some idx in this manner.
    def subset_cmp(ss1, ss2, direction, start, v):
        if ss1 is None or ss2 is None:
            return True
        idx = start
        while idx <= 2 and idx >= 0:
            v1 = get_value(idx, ss1, v)
            v2 = get_value(idx, ss2, v)
            if v1 != v2:
                return v1 > v2
            idx += direction
        return True

    # Compare the value on agent at start, then start + direction, etc.
    # Return true if ss1 is less than ss2 at start or greater than ss2 after start.
    def subset_cmp_lowest_give_away(ss1, ss2, direction, start, v):
        if ss1 is None or ss2 is None:
            return True
        idx = start
        # while idx <= 2 and idx >= 0:
        #     v1 = get_value(idx, ss1, v)
        #     v2 = get_value(idx, ss2, v)
        #     if v1 != v2 and idx == start:
        #         return v1 < v2
        #     elif v1 != v2:
        #         return v1 > v2
        #     idx += direction
        # return True
        v1 = get_value(idx, ss1, v)
        v2 = get_value(idx, ss2, v)
        return v1 <= v2

    # Transfer the subset from a1 to a2 in allocation _alloc
    def flow_subset(_alloc, a1, a2, subset):
        flowed_alloc = deepcopy(_alloc)
        for g in subset:
            flowed_alloc[a1].remove(g)
            flowed_alloc[a2].append(g)
        return flowed_alloc

    # Try to transfer some subset of goods from a1 to a2 to make them pairwise EFX.
    # _fixed_good must stay with a1.
    def flow(_alloc, a1, a2, _fixed_good):
        if _alloc is None:
            return None

        pair_G = nx.Graph()
        pair_G.add_nodes_from([a1, a2])
        pair_G.add_edge(a1, a2)

        if alloc_is_pefx(_alloc, pair_G, val_fns):
            return _alloc

        chosen_subset = None
        for subset in powerset(_alloc[a1]):
            if _fixed_good not in subset:
                # if subset_cmp(subset, chosen_subset, a2-a1, a2, val_fns):
                if subset_cmp_lowest_give_away(subset, chosen_subset, a2 - a1, a1, val_fns):
                    flowed_alloc = flow_subset(_alloc, a1, a2, subset)

                    # if alloc_is_mms(flowed_alloc, pair_G, val_fns):
                    if alloc_is_pefx(flowed_alloc, pair_G, val_fns):
                        chosen_subset = deepcopy(subset)

        if chosen_subset is not None:
            flowed_alloc = flow_subset(_alloc, a1, a2, chosen_subset)
            return flowed_alloc
        else:
            return None

    def assign_left(alloc, G, val_fns, good, init_alloc):
        _alloc = deepcopy(alloc)
        _alloc[0].append(good)
        # Flow right
        while not alloc_is_pefx(_alloc, G, val_fns):
            prev_alloc = deepcopy(_alloc)
            # _alloc = flow(_alloc, 0, 1, good)
            _alloc = flow(_alloc, 0, 1, None)

            if _alloc is None:
                _alloc = init_alloc
                break
            if not alloc_is_pefx(_alloc, G, val_fns):
                # _alloc = flow(_alloc, 1, 2, good)
                _alloc = flow(_alloc, 1, 2, None)

            if prev_alloc == _alloc or _alloc is None:
                _alloc = init_alloc
                break
        return _alloc

    def assign_right(alloc, G, val_fns, good, init_alloc):
        _alloc = deepcopy(alloc)
        _alloc[2].append(good)
        # Flow right
        while not alloc_is_pefx(_alloc, G, val_fns):
            prev_alloc = deepcopy(_alloc)
            # _alloc = flow(_alloc, 2, 1, good)
            _alloc = flow(_alloc, 2, 1, None)

            if _alloc is None:
                _alloc = init_alloc
                break
            if not alloc_is_pefx(_alloc, G, val_fns):
                # _alloc = flow(_alloc, 1, 0, good)
                _alloc = flow(_alloc, 1, 0, None)

            if prev_alloc == _alloc or _alloc is None:
                _alloc = init_alloc
                break
        return _alloc

    def assign_center(alloc, G, val_fns, good, init_alloc):
        _alloc = deepcopy(alloc)
        _alloc[1].append(good)

        while not alloc_is_pefx(_alloc, G, val_fns):
            preflow = deepcopy(_alloc)
            _alloc = flow(_alloc, 1, 0, None)

            if _alloc is None:
                _alloc = init_alloc
                break
            if not alloc_is_pefx(_alloc, G, val_fns):
                # _alloc = flow(_alloc, 1, 0, good)
                _alloc = flow(_alloc, 1, 2, None)

            if preflow == _alloc or _alloc is None:
                _alloc = init_alloc
                break

            # if _alloc is None:
            #     _alloc = preflow
            #     _alloc = [_alloc[1], _alloc[0], _alloc[2]]

            # preflow = deepcopy(_alloc)
            # _alloc = flow(_alloc, 1, 2, good)
            # _alloc = flow(_alloc, 1, 2, None)

            # if _alloc is None:
            #     _alloc = preflow
            #     _alloc = [_alloc[0], _alloc[2], _alloc[1]]

            # if not alloc_is_pefx(_alloc, G, val_fns):
            #     print("Algorithm doesn't work")
            #     return None

        # if not alloc_is_pefx(_alloc, G, val_fns):
        #     if alloc_is_pefx([_alloc[0], _alloc[2], _alloc[1]], G, val_fns) and \
        #             get_value(1, _alloc[2], val_fns) >= get_value(1, _alloc[1], val_fns) and \
        #             get_value(2, _alloc[1], val_fns) >= get_value(2, _alloc[2], val_fns):
        #         _alloc = [_alloc[0], _alloc[2], _alloc[1]]
        #     elif alloc_is_pefx([_alloc[1], _alloc[0], _alloc[2]], G, val_fns) and \
        #             get_value(1, _alloc[0], val_fns) >= get_value(1, _alloc[1], val_fns) and \
        #             get_value(0, _alloc[1], val_fns) >= get_value(0, _alloc[0], val_fns):
        #         _alloc = [_alloc[1], _alloc[0], _alloc[2]]
        #     else:
        #         preflow = deepcopy(_alloc)
        #         # _alloc = flow(_alloc, 1, 0, good)
        #         _alloc = flow(_alloc, 1, 0, None)
        #
        #         if _alloc is None:
        #             _alloc = preflow
        #             _alloc = [_alloc[1], _alloc[0], _alloc[2]]
        #
        #         preflow = deepcopy(_alloc)
        #         # _alloc = flow(_alloc, 1, 2, good)
        #         _alloc = flow(_alloc, 1, 2, None)
        #
        #         if _alloc is None:
        #             _alloc = preflow
        #             _alloc = [_alloc[0], _alloc[2], _alloc[1]]
        #
        #         if not alloc_is_pefx(_alloc, G, val_fns):
        #             print("Algorithm doesn't work")
        #             return None

        return _alloc

    def leximin_value(alloc, val_fns):
        return sorted([get_value(i, alloc[i], val_fns) for i in range(3)])


    def restricted_po(alloc, val_fns):
        valid_items = set(alloc[0]) | set(alloc[1]) | set(alloc[2])
        new_val_fns = {0: {}, 1: {}, 2: {}}
        for a in val_fns:
            for g in valid_items:
                new_val_fns[a][g] = val_fns[a][g]
        return po(alloc, new_val_fns)

    # create allocation
    alloc = [[], [], []]
    # map from goods to order fn
    good_to_order_fn = {}
    for g in goods:
        # good_to_order_fn[g] = sorted([val_fns[i][g] for i in range(3)], reverse=True)
        good_to_order_fn[g] = sorted([val_fns[i][g] for i in range(3)], reverse=True)
    goods_ordered = sorted(goods, key=lambda x: good_to_order_fn[x], reverse=Truep)
    # goods_ordered = sorted(goods, key=lambda x: random.random())

    # Try agents in the order that they value the good
    # assign the good, check for pefx after flowing. If not, move to the next agent in the order
    for good in goods_ordered:
        init_alloc = deepcopy(alloc)

        a0 = assign_left(alloc, G, val_fns, good, init_alloc)
        a1 = assign_center(alloc, G, val_fns, good, init_alloc)
        a2 = assign_right(alloc, G, val_fns, good, init_alloc)

        # if a0 == init_alloc:
        #     print(alloc)
        #     print(val_fns)
        #     print(a2)
        #     print("\n\n\n")

        allocs = sorted([a0, a1, a2], key=lambda x: leximin_value(x, val_fns), reverse=True)
        # allocs = sorted([a0, a2], key=lambda x: leximax_value(x, val_fns), reverse=True)

        for a in allocs:
            # if a and a != init_alloc and alloc_is_pefx(a, G, val_fns) and restricted_po(a, val_fns):
            if a and a != init_alloc and alloc_is_pefx(a, G, val_fns):
                alloc = a
                break

        if alloc == init_alloc:
            print(good)
            print(alloc)
            return None
        # agent_order = np.argsort([val_fns[i][good] for i in range(3)])[::-1]
        # for a in agent_order:
        #     if a == 0:
        #         alloc = assign_left(alloc, G, val_fns, good, init_alloc)
        #     elif a == 2:
        #         alloc = assign_right(alloc, G, val_fns, good, init_alloc)
        #     elif a == 1:
        #         alloc = assign_center(alloc, G, val_fns, good, init_alloc)
        #
        #     if alloc != init_alloc:
        #         # We successfully assigned the good and got a p-efx allocation, move on
        #         break

    if not alloc_is_pefx(alloc, G, val_fns):
        return None
    else:
        return alloc


def yairs_algo_pairwise_lm(G, val_fns):
    goods = list(range(len(val_fns[0].values())))

    # Compare the value on agent at start, then start + direction, etc.
    # Return true if ss1 is greater than ss2 at some idx in this manner.
    def subset_cmp(ss1, ss2, direction, start, v):
        if ss1 is None or ss2 is None:
            return True
        idx = start
        while idx <= 2 and idx >= 0:
            v1 = get_value(idx, ss1, v)
            v2 = get_value(idx, ss2, v)
            if v1 != v2:
                return v1 > v2
            idx += direction
        return True

    # Compare the value on agent at start, then start + direction, etc.
    # Return true if ss1 is less than ss2 at start or greater than ss2 after start.
    def subset_cmp_lowest_give_away(ss1, ss2, direction, start, v):
        if ss1 is None or ss2 is None:
            return True
        idx = start
        # while idx <= 2 and idx >= 0:
        #     v1 = get_value(idx, ss1, v)
        #     v2 = get_value(idx, ss2, v)
        #     if v1 != v2 and idx == start:
        #         return v1 < v2
        #     elif v1 != v2:
        #         return v1 > v2
        #     idx += direction
        # return True
        v1 = get_value(idx, ss1, v)
        v2 = get_value(idx, ss2, v)
        return v1 <= v2

    # Transfer the subset from a1 to a2 in allocation _alloc
    def flow_subset(_alloc, a1, a2, subset):
        flowed_alloc = deepcopy(_alloc)
        for g in subset:
            flowed_alloc[a1].remove(g)
            flowed_alloc[a2].append(g)
        return flowed_alloc

    # Try to transfer some subset of goods from a1 to a2 to make them pairwise EFX.
    # _fixed_good must stay with a1.
    def flow(_alloc, a1, a2, _fixed_good):
        if _alloc is None:
            return None

        pair_G = nx.Graph()
        pair_G.add_nodes_from([a1, a2])
        pair_G.add_edge(a1, a2)

        if is_pairwise_leximin(_alloc, pair_G, val_fns):
            return _alloc

        chosen_subset = None
        for subset in powerset(_alloc[a1]):
            if _fixed_good not in subset:
                # if subset_cmp(subset, chosen_subset, a2-a1, a2, val_fns):
                if subset_cmp_lowest_give_away(subset, chosen_subset, a2 - a1, a1, val_fns):
                    flowed_alloc = flow_subset(_alloc, a1, a2, subset)

                    # if alloc_is_mms(flowed_alloc, pair_G, val_fns):
                    if is_pairwise_leximin(flowed_alloc, pair_G, val_fns):
                        chosen_subset = deepcopy(subset)

        if chosen_subset is not None:
            flowed_alloc = flow_subset(_alloc, a1, a2, chosen_subset)
            return flowed_alloc
        else:
            return None

    def assign_left(alloc, G, val_fns, good, init_alloc):
        _alloc = deepcopy(alloc)
        _alloc[0].append(good)
        # Flow right
        while not is_pairwise_leximin(_alloc, G, val_fns):
            prev_alloc = deepcopy(_alloc)
            # _alloc = flow(_alloc, 0, 1, good)
            _alloc = flow(_alloc, 0, 1, None)

            if _alloc is None:
                _alloc = init_alloc
                break
            if not is_pairwise_leximin(_alloc, G, val_fns):
                # _alloc = flow(_alloc, 1, 2, good)
                _alloc = flow(_alloc, 1, 2, None)

            if prev_alloc == _alloc or _alloc is None:
                _alloc = init_alloc
                break
        return _alloc

    def assign_right(alloc, G, val_fns, good, init_alloc):
        _alloc = deepcopy(alloc)
        _alloc[2].append(good)
        # Flow right
        while not is_pairwise_leximin(_alloc, G, val_fns):
            prev_alloc = deepcopy(_alloc)
            # _alloc = flow(_alloc, 0, 1, good)
            _alloc = flow(_alloc, 2, 1, None)

            if _alloc is None:
                _alloc = init_alloc
                break
            if not is_pairwise_leximin(_alloc, G, val_fns):
                # _alloc = flow(_alloc, 1, 2, good)
                _alloc = flow(_alloc, 1, 0, None)

            if prev_alloc == _alloc or _alloc is None:
                _alloc = init_alloc
                break
        return _alloc

    def leximax_value(alloc, val_fns):
        return sorted([get_value(i, alloc[i], val_fns) for i in range(3)])

    def restricted_po(alloc, val_fns):
        valid_items = set(alloc[0]) | set(alloc[1]) | set(alloc[2])
        new_val_fns = {0: {}, 1: {}, 2: {}}
        for a in val_fns:
            for g in valid_items:
                new_val_fns[a][g] = val_fns[a][g]
        return po(alloc, new_val_fns)

    # create allocation
    alloc = [[], [], []]
    # map from goods to order fn
    good_to_order_fn = {}
    # for g in goods:
    #     good_to_order_fn[g] = sorted([val_fns[i][g] for i in range(3)], reverse=True)
    # goods_ordered = sorted(goods, key=lambda x: good_to_order_fn[x])
    goods_ordered = sorted(goods, key=lambda x: random.random())

    # Try agents in the order that they value the good
    # assign the good, check for pefx after flowing. If not, move to the next agent in the order
    for good in goods_ordered:
        init_alloc = deepcopy(alloc)

        a0 = assign_left(alloc, G, val_fns, good, init_alloc)
        # a1 = assign_center(alloc, G, val_fns, good, init_alloc)
        a2 = assign_right(alloc, G, val_fns, good, init_alloc)

        # if a0 == init_alloc:
        #     print(alloc)
        #     print(val_fns)
        #     print(a2)
        #     print("\n\n\n")

        # allocs = sorted([a0, a1, a2], key=lambda x: leximax_value(x, val_fns), reverse=True)
        allocs = sorted([a0, a2], key=lambda x: leximax_value(x, val_fns), reverse=True)

        for a in allocs:
            # if a and a != init_alloc and alloc_is_pefx(a, G, val_fns) and restricted_po(a, val_fns):
            if a and a != init_alloc and is_pairwise_leximin(a, G, val_fns):
                alloc = a
                break

        if alloc == init_alloc:
            return None
        # agent_order = np.argsort([val_fns[i][good] for i in range(3)])[::-1]
        # for a in agent_order:
        #     if a == 0:
        #         alloc = assign_left(alloc, G, val_fns, good, init_alloc)
        #     elif a == 2:
        #         alloc = assign_right(alloc, G, val_fns, good, init_alloc)
        #     elif a == 1:
        #         alloc = assign_center(alloc, G, val_fns, good, init_alloc)
        #
        #     if alloc != init_alloc:
        #         # We successfully assigned the good and got a p-efx allocation, move on
        #         break

    if not is_pairwise_leximin(alloc, G, val_fns):
        return None
    else:
        return alloc


def test_get_all_allocs():
    test_agents = [0, 1, 2]
    test_goods = [0, 1, 2]
    test_heur = list(get_all_allocs_with_heuristic(test_agents, test_goods))
    test_all = list(get_all_allocs(test_agents, test_goods))
    print(len(test_heur) == len(set(test_heur)))
    print(len(test_all) == len(set(test_all)))
    print(set(test_heur))
    print(">>>>>>>>>>>>>>>>")
    print(set(test_all))
    print(len(set(test_all)))
    print(">>>>>>>>>>>>>>>>\n")
    print(len(set(test_all) - set(test_heur)))
    print(">>>>>>>>>>>>>>>>\n")

    print(list(get_all_partitions(set(test_goods), 3, 3)))
    print(list(powerset(range(3))))


if __name__ == "__main__":
    # print(find_po_and_efx_alloc_cgm20())
    # print(run_exhaustive_sim(list(range(5)), compute_po_efx))

    random.seed(1)
    for i in range(int(1e9)):
        if not rand_sim(list(range(5)), yairs_algo):
            sys.exit(0)

    # random.seed(15)
    # print(run_exhaustive_sim(list(range(6)), yairs_algo_pairwise_lm))
    # {0: {0: 7, 1: 6, 2: 20, 3: 14, 4: 7}, 1: {0: 13, 1: 18, 2: 16, 3: 6, 4: 20}, 2: {0: 3, 1: 16, 2: 8, 3: 7, 4: 3}}

    # vf = {0: {0: 8, 1: 5, 2: 4, 3: 13, 4: 2},
    #       1: {0: 6, 1: 3, 2: 4, 3: 15, 4: 18},
    #       2: {0: 15, 1: 1, 2: 2, 3: 9, 4: 2}}
    # print(run_sim(vf, compute_po_efx))
