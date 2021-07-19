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
    for other_alloc in get_all_allocs(list(range(len(alloc))), list(range(len(val_fns[0])))):
        if pareto_dominates(other_alloc, alloc, val_fns):
            return False
    return True


def mms(a1, a2, alloc, val_fns):
    all_goods = list(alloc[a1])
    all_goods.extend(list(alloc[a2]))
    mms1 = 0
    mms2 = 0
    for split in get_splits(all_goods):
        min_a1 = min(get_value(a1, split[0], val_fns), get_value(a1, split[1], val_fns))
        if min_a1 > mms1:
            mms1 = min_a1
        min_a2 = min(get_value(a2, split[0], val_fns), get_value(a2, split[1], val_fns))
        if min_a2 > mms2:
            mms2 = min_a2
    return get_value(a1, alloc[a1], val_fns) >= mms1 and get_value(a2, alloc[a2], val_fns) >= mms2


def alloc_is_mms(alloc, G, val_fns):
    for e in G.edges():
        if not mms(e[0], e[1], alloc, val_fns):
            return False
    return True


def find_pairwise_mms_alloc(G, val_fns, agents, goods):
    for alloc in get_all_allocs_with_heuristic(agents, goods):
        # print(alloc)
        if alloc_is_mms(alloc, G, val_fns):
            print("SUCCESS")
            return alloc
        # print("NO")
    print("FAILURE IN HEURISTIC SEARCH")
    # for alloc in get_all_allocs(agents, goods):
    #     # print(alloc)
    #     if alloc_is_mms(alloc, G, val_fns):
    #         print("SUCCESS")
    #         return alloc
    print("FAIL")


def run_sim(agents, goods):
    G = nx.Graph()
    G.add_nodes_from(agents)
    for p in combinations(agents, 2):
        if random.random() < 0.7:
            try:
                G.add_edge(p[0], p[1])
                nx.find_cycle(G, orientation="ignore")  # Throws if cycle not found
                G.remove_edge(p[0], p[1])
            except:
                pass

    val_fns = {}
    for a in agents:
        val_fns[a] = {}
        for g in goods:
            val_fns[a][g] = random.random()

    print(G.edges())
    print(val_fns)
    if not find_pairwise_mms_alloc(G, val_fns, agents, goods):
        print("FAILED TO FIND PMMS ALLOCATION")
        sys.exit(0)
    print("\n***********\n***********\n***********\n")


def run_sim_star(agents, goods):
    G = nx.Graph()
    G.add_nodes_from(agents)
    for a in agents[1:]:
        G.add_edge(agents[0], a)
    # for p in combinations(agents, 2):
    #     if random.random() < 0.7:
    #         try:
    #             G.add_edge(p[0], p[1])
    #             nx.find_cycle(G, orientation="ignore") # Throws if cycle not found
    #             G.remove_edge(p[0], p[1])
    #         except:
    #             pass

    val_fns = {}
    for a in agents:
        val_fns[a] = {}
        for g in goods:
            val_fns[a][g] = random.random()

    print(G.edges())
    print(val_fns)
    if not find_pairwise_mms_alloc(G, val_fns, agents, goods):
        print("FAILED TO FIND PMMS ALLOCATION")
        sys.exit(0)
    print("\n***********\n***********\n***********\n")


# We had a running algorithm idea that we would give goods in order of their max value,
# giving them to the agent that valued them the most. Then we hoped that when you give
# to an "edge" agent in the triplet on a line, you would allow the edge agent to keep the new
# good no matter what. This is not always possible.

# Transfers items in sorted order of v_i/v_j from i to j. Lets i keep the new good that
# is more valuable than all the others, and is more valuable to i than j.
# This is not always possible!!!!!!!!!!!!!
def transfer_for_pmms(G, val_fns, goods):
    print("begin transfer_for_pmms")
    # create allocation
    g_prime = deepcopy(goods)
    alloc = [g_prime, []]
    # map from goods to ratio
    good_to_ratio = {}
    for g in goods:
        if val_fns[1][g] > 0:
            good_to_ratio[g] = val_fns[0][g] / val_fns[1][g]
        else:
            good_to_ratio[g] = 1e10
    goods_ordered = sorted(goods, key=lambda x: good_to_ratio[x])
    # transfer goods and check for pmms
    for good in goods_ordered:
        if good != len(goods) - 1:
            alloc[0].remove(good)
            alloc[1].append(good)
            print("testing alloc " + str(alloc))
            if alloc_is_mms(alloc, G, val_fns):
                return alloc
    return None


def one_way_flow_pmms(G, val_fns, goods):
    # print("begin one_way_flow_pmms")
    # create allocation
    alloc = [[], []]
    # map from goods to ratio
    good_to_ratio = {}
    for g in goods:
        if val_fns[1][g] > 0:
            good_to_ratio[g] = val_fns[0][g] / val_fns[1][g]
        else:
            good_to_ratio[g] = 1e10
    goods_ordered = sorted(goods, key=lambda x: good_to_ratio[x])
    # assign a good, check for pmms, if no, give to other agent
    for good in goods_ordered:
        alloc[0].append(good)
        if not alloc_is_mms(alloc, G, val_fns):
            alloc[0].remove(good)
            alloc[1].append(good)
            if not alloc_is_mms(alloc, G, val_fns):
                print("MAJOR PROBLEM")
                sys.exit(0)
    if not alloc_is_mms(alloc, G, val_fns):
        return None
    else:
        return alloc


def one_way_flow_allowable_pmms(G, val_fns, goods):
    # print("begin one_way_flow_pmms")
    # create allocation
    alloc = [[], []]
    # map from goods to ratio
    good_to_ratio = {}
    for g in goods:
        if val_fns[1][g] > 0:
            good_to_ratio[g] = val_fns[0][g] / val_fns[1][g]
        else:
            good_to_ratio[g] = 1e10
    goods_ordered = sorted(goods, key=lambda x: good_to_ratio[x])
    # assign a good, check for pmms, if no, give to other agent
    for good in goods_ordered:
        alloc[0].append(good)
        print(alloc)
        if not alloc_is_mms(alloc, G, val_fns):
            for g in sorted(alloc[0], key=lambda x: good_to_ratio[x]):
                print("moving " + str(g))
                test_alloc = deepcopy(alloc)
                test_alloc[0].remove(g)
                test_alloc[1].append(g)
                if alloc_is_mms(test_alloc, G, val_fns):
                    print(test_alloc)
                    alloc = test_alloc
                    break
    if not alloc_is_mms(alloc, G, val_fns):
        return None
    else:
        return alloc


def one_way_flow_by_ratio_pmms(G, val_fns, goods):
    # print("begin one_way_flow_pmms")
    # create allocation
    alloc = [[], []]
    # map from goods to ratio
    good_to_ratio = {}
    for g in goods:
        if val_fns[1][g] > 0:
            good_to_ratio[g] = val_fns[0][g] / val_fns[1][g]
        else:
            good_to_ratio[g] = 1e10
    goods_ordered = sorted(goods, key=lambda x: good_to_ratio[x])
    # assign a good, check for pmms, if no, give to other agent
    for good in goods_ordered:
        alloc[0].append(good)
        # print(alloc)
        if not alloc_is_mms(alloc, G, val_fns):
            for g in sorted(alloc[0], key=lambda x: good_to_ratio[x]):
                # print("moving " + str(g))
                test_alloc = deepcopy(alloc)
                test_alloc[0].remove(g)
                test_alloc[1].append(g)
                if alloc_is_mms(test_alloc, G, val_fns):
                    # print(test_alloc)
                    alloc = test_alloc
                    break
    if not alloc_is_mms(alloc, G, val_fns):
        return None
    else:
        return alloc


def find_any_xfers_one_way_flow_pmms_top(G, val_fns, goods, order_fn):
    return find_any_xfers_one_way_flow_pmms_recursive(G, val_fns, set(goods), [[], []], order_fn)


def find_any_xfers_one_way_flow_pmms_recursive(G, val_fns, goods, alloc, order_fn):
    if len(goods) == 0:
        return alloc
    goods_ordered = sorted(goods, key=order_fn)
    # assign a good, check for pmms, if no, see if there is any set of items to give the other agent such that
    # the transfer is irreversible and we can finish with a pmms allocation.
    for good in goods_ordered:
        alloc[0].append(good)
        if not alloc_is_mms(alloc, G, val_fns):
            # Determine if there is any good to irreversibly transfer which will end up in a PMMS allocation
            for g in alloc[0]:
                test_alloc = deepcopy(alloc)
                test_alloc[0].remove(g)
                test_alloc[1].append(g)
                if alloc_is_mms(test_alloc, G, val_fns):
                    recursive_soln = find_any_xfers_one_way_flow_pmms_recursive(G, val_fns,
                                                                                goods - set(test_alloc[0]) - set(
                                                                                    test_alloc[1]),
                                                                                test_alloc, order_fn)
                    if recursive_soln:
                        return recursive_soln

    if not alloc_is_mms(alloc, G, val_fns):
        return None
    else:
        return alloc


def yairs_algo(G, val_fns, goods):
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
        while idx <= 2 and idx >= 0:
            v1 = get_value(idx, ss1, v)
            v2 = get_value(idx, ss2, v)
            if v1 != v2 and idx == start:
                return v1 < v2
            elif v1 != v2:
                return v1 > v2
            idx += direction
        return True

    # Transfer the subset from a1 to a2 in allocation _alloc
    def flow_subset(_alloc, a1, a2, subset):
        flowed_alloc = deepcopy(_alloc)
        for g in subset:
            flowed_alloc[a1].remove(g)
            flowed_alloc[a2].append(g)
        return flowed_alloc

    # Try to transfer some subset of goods from a1 to a2 to make them pairwise MMS.
    # _fixed_good must stay with a1.
    def flow(_alloc, a1, a2, _fixed_good):
        if _alloc is None:
            return None

        pair_G = nx.Graph()
        pair_G.add_nodes_from([a1, a2])
        pair_G.add_edge(a1, a2)

        if alloc_is_mms(_alloc, pair_G, val_fns):
            return _alloc

        chosen_subset = None
        for subset in powerset(_alloc[a1]):
            if _fixed_good not in subset:
                # if subset_cmp(subset, chosen_subset, a2-a1, a2, val_fns):
                if subset_cmp_lowest_give_away(subset, chosen_subset, a2 - a1, a1, val_fns):
                    flowed_alloc = flow_subset(_alloc, a1, a2, subset)

                    if alloc_is_mms(flowed_alloc, pair_G, val_fns):
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
        while not alloc_is_mms(_alloc, G, val_fns):
            prev_alloc = deepcopy(_alloc)
            # _alloc = flow(_alloc, 0, 1, good)
            _alloc = flow(_alloc, 0, 1, None)

            if _alloc is None:
                _alloc = init_alloc
                break
            if not alloc_is_mms(_alloc, G, val_fns):
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
        while not alloc_is_mms(_alloc, G, val_fns):
            prev_alloc = deepcopy(_alloc)
            # _alloc = flow(_alloc, 2, 1, good)
            _alloc = flow(_alloc, 2, 1, None)

            if _alloc is None:
                _alloc = init_alloc
                break
            if not alloc_is_mms(_alloc, G, val_fns):
                # _alloc = flow(_alloc, 1, 0, good)
                _alloc = flow(_alloc, 1, 0, None)

            if prev_alloc == _alloc or _alloc is None:
                _alloc = init_alloc
                break
        return _alloc

    # create allocation
    alloc = [[], [], []]
    # map from goods to order fn
    good_to_order_fn = {}
    for g in goods:
        good_to_order_fn[g] = sorted([val_fns[i][g] for i in range(3)], reverse=True)
    goods_ordered = sorted(goods, key=lambda x: good_to_order_fn[x])
    # assign a good, check for pmms, if no, give to other agent
    for good in goods_ordered:
        init_alloc = deepcopy(alloc)
        tried_assigning = None
        if val_fns[0][good] >= val_fns[2][good]:
            alloc = assign_left(alloc, G, val_fns, good, init_alloc)
            tried_assigning = 0
        elif val_fns[0][good] < val_fns[2][good]:
            alloc = assign_right(alloc, G, val_fns, good, init_alloc)
            tried_assigning = 2
        # elif alloc_is_mms([alloc[0] + [good], alloc[1], alloc[2]], G, val_fns):
        #     alloc[0].append(good)
        # elif alloc_is_mms([alloc[0], alloc[1], alloc[2] + [good]], G, val_fns):
        #     alloc[2].append(good)
        # else:
        #     alloc = assign_left(alloc, G, val_fns, good, init_alloc)
        #     tried_assigning = 0

        # Try the other side
        if alloc == init_alloc:
            if tried_assigning == 0:
                alloc = assign_right(alloc, G, val_fns, good, init_alloc)
            else:
                alloc = assign_left(alloc, G, val_fns, good, init_alloc)

        # If we failed above, that means that we need to add the new good to the center agent
        if alloc == init_alloc:
            alloc[1].append(good)

            if not alloc_is_mms(alloc, G, val_fns):
                if alloc_is_mms([alloc[0], alloc[2], alloc[1]], G, val_fns) and \
                        get_value(1, alloc[2], val_fns) >= get_value(1, alloc[1], val_fns) and \
                        get_value(2, alloc[1], val_fns) >= get_value(2, alloc[2], val_fns):
                    alloc = [alloc[0], alloc[2], alloc[1]]
                elif alloc_is_mms([alloc[1], alloc[0], alloc[2]], G, val_fns) and \
                        get_value(1, alloc[0], val_fns) >= get_value(1, alloc[1], val_fns) and \
                        get_value(0, alloc[1], val_fns) >= get_value(0, alloc[0], val_fns):
                    alloc = [alloc[1], alloc[0], alloc[2]]
                else:
                    preflow = deepcopy(alloc)
                    # alloc = flow(alloc, 1, 0, good)
                    alloc = flow(alloc, 1, 0, None)

                    if alloc is None:
                        alloc = preflow
                        alloc = [alloc[1], alloc[0], alloc[2]]

                    preflow = deepcopy(alloc)
                    # alloc = flow(alloc, 1, 2, good)
                    alloc = flow(alloc, 1, 2, None)

                    if alloc is None:
                        alloc = preflow
                        alloc = [alloc[0], alloc[2], alloc[1]]

                    if not alloc_is_mms(alloc, G, val_fns):
                        print("Algorithm doesn't work")
                        return None

    if not alloc_is_mms(alloc, G, val_fns):
        return None
    else:
        return alloc


def run_sim_pair(goods, updated_alloc_algo):
    agents = [0, 1]
    G = nx.Graph()
    G.add_nodes_from(agents)
    G.add_edge(agents[0], agents[1])

    val_fns = {}
    for a in agents:
        val_fns[a] = {}
        for g in goods:
            val_fns[a][g] = random.randint(1 + (a) * 5, 10 - (1 - a) * 5)

    print(val_fns)
    # alloc = find_pairwise_mms_alloc(G, val_fns, agents, goods)
    # if not alloc:
    #     print("FAILED TO FIND INITIAL PMMS ALLOCATION")
    #     sys.exit(0)
    # else:
    #     print("INITIAL PMMS EXISTS")
    #     print(alloc)

    # g = len(goods)
    # goods.append(g)
    # best_value = max(max(val_fns[0].values()), max(val_fns[1].values()))
    # val_fns[0][g] = best_value + random.randint(0, 5)
    # val_fns[1][g] = max(random.randint(0, 10), val_fns[0][1])

    print(val_fns)

    new_alloc = updated_alloc_algo(G, val_fns, goods)

    if not new_alloc:
        print(str(updated_alloc_algo) + " FAILED")
        sys.exit(0)
    else:
        print(str(updated_alloc_algo) + " SUCCESSFUL")
        print(new_alloc)

    print("\n***********\n***********\n***********\n")


def run_sim_triplet(goods, alloc_algo):
    agents = [0, 1, 2]
    G = nx.Graph()
    G.add_nodes_from(agents)
    G.add_edge(agents[0], agents[1])
    G.add_edge(agents[1], agents[2])

    val_fns = {}
    for a in agents:
        val_fns[a] = {}
        for g in goods:
            val_fns[a][g] = random.randint(1 + (a) * 5, 15 - (1 - a) * 5)

    print(val_fns)

    alloc = alloc_algo(G, val_fns, goods)

    if not alloc:
        print(str(alloc_algo) + " FAILED")
        sys.exit(0)
    else:
        print(str(alloc_algo) + " SUCCESSFUL")
        print(alloc)

    print("\n***********\n***********\n***********\n")


def run_exhaustive_sim_pair(goods, updated_alloc_algo):
    agents = [0, 1]
    max_val = 7
    G = nx.Graph()
    G.add_nodes_from(agents)
    G.add_edge(agents[0], agents[1])

    val_fns = {}
    val_fns[0] = {}

    r = [range(1, max_val)] * (2 * len(goods))
    for v in product(*r):
        old_a1 = deepcopy(val_fns[0])

        i = 0
        for a in agents:
            val_fns[a] = {}
            for g in goods:
                val_fns[a][g] = v[i]
                i += 1

        if val_fns[0] != old_a1:
            print(val_fns)

        new_alloc = updated_alloc_algo(G, val_fns, goods)

        if not new_alloc:
            print(val_fns)
            print(str(updated_alloc_algo) + " FAILED")
            sys.exit(0)
        # else:
        #     print(str(updated_alloc_algo) + " SUCCESSFUL")
        #     print(new_alloc)

        # print("\n***********\n***********\n***********\n")


def run_exhaustive_sim_triplet(goods, alloc_algo):
    # Returns true if there are duplicates according to this set of valuations
    def contains_dup_goods(v):
        goods = set()
        for i in range(len(v[0])):
            good = sorted([v[0][i], v[1][i], v[2][i]])
            good = (good[0], good[1], good[2])
            if good in goods:
                return True
            goods.add(good)
        return False

    # Returns true if there are duplicates according to this set of valuations within any pair of agents
    def contains_dup_goods_pairwise(v):
        good_pairs = set()
        for i in range(len(v[0])):
            for good_pair in [[0, 1] + [v[0][i], v[1][i]],
                              [0, 2] + [v[0][i], v[2][i]],
                              [1, 2] + [v[1][i], v[2][i]]]:
                good_pair = (good_pair[0], good_pair[1], good_pair[2], good_pair[3])
                if good_pair in good_pairs:
                    return True
                good_pairs.add(good_pair)
        return False


    agents = [0, 1, 2]
    # val_options = list(range(1,7))
    val_options = [random.randint(1, 50) for _ in range(4)]

    G = nx.Graph()
    G.add_nodes_from(agents)
    G.add_edge(agents[0], agents[1])
    G.add_edge(agents[1], agents[2])

    val_fns = {}
    val_fns[0] = {}

    r = [val_options] * (3 * len(goods))
    for v in product(*r):
        old_a1 = deepcopy(val_fns[0])

        i = 0
        for a in agents:
            val_fns[a] = {}
            for g in goods:
                val_fns[a][g] = v[i]
                i += 1

        if val_fns[0] != old_a1:
            print(val_fns)

        if not contains_dup_goods_pairwise(val_fns) and not contains_dup_goods(val_fns) and random.random() < .01:
        # if random.random() < .01:
            try:
                new_alloc = alloc_algo(G, val_fns, goods)
            except Exception as e:
                print(val_fns)
                print(e)

            if not new_alloc:
                print(val_fns)
                print(str(alloc_algo) + " FAILED")
                sys.exit(0)
            else:
                if not po(new_alloc, val_fns):
                    print("Not PO")
                print(str(alloc_algo) + " SUCCESSFUL")
                print(new_alloc)

            # print("\n***********\n***********\n***********\n")


if __name__ == "__main__":
    # G = nx.Graph()
    # G.add_nodes_from([0,1,2])
    # G.add_edges_from([(0,1), (1,2), (0,2)])
    #
    # agents = [0,1,2]
    # goods = list(range(12))
    #
    # val_fns = {0: {0: 1e6 + 17*1e3, 1: 1e6 + 25*1e3, 2: 1e6 + 12*1e3, 3: 1e6 + 1*1e3,
    #                4: 1e6 + 2*1e3, 5: 1e6 + 22*1e3, 6: 1e6 + 3*1e3, 7: 1e6 + 28*1e3,
    #                8: 1e6 + 11*1e3, 9: 1e6 + 0*1e3, 10: 1e6 + 21*1e3, 11: 1e6 + 23*1e3},
    # 1: {0: 1e6 + 17*1e3, 1: 1e6 + 25*1e3, 2: 1e6 + 12*1e3, 3: 1e6 + 1*1e3,
    #                4: 1e6 + 2*1e3, 5: 1e6 + 22*1e3, 6: 1e6 + 3*1e3, 7: 1e6 + 28*1e3,
    #                8: 1e6 + 11*1e3, 9: 1e6 + 0*1e3, 10: 1e6 + 21*1e3, 11: 1e6 + 23*1e3},
    # 2: {0: 1e6 + 17*1e3, 1: 1e6 + 25*1e3, 2: 1e6 + 12*1e3, 3: 1e6 + 1*1e3,
    #                4: 1e6 + 2*1e3, 5: 1e6 + 22*1e3, 6: 1e6 + 3*1e3, 7: 1e6 + 28*1e3,
    #                8: 1e6 + 11*1e3, 9: 1e6 + 0*1e3, 10: 1e6 + 21*1e3, 11: 1e6 + 23*1e3}}
    # val_fns[0][0] += 3
    # val_fns[0][1] -= 1
    # val_fns[0][2] -= 1
    # val_fns[0][3] -= 1
    #
    # val_fns[1][0] += 3
    # val_fns[1][1] -= 1
    # val_fns[1][4] -= 1
    # val_fns[1][8] -= 1
    #
    # val_fns[2][0] += 3
    # val_fns[2][2] -= 1
    # val_fns[2][6] -= 1
    # val_fns[2][11] -= 1
    #
    # print(find_pairwise_mms_alloc(G, val_fns, agents, goods))

    # random.seed(14)
    # for i in range(100000):
    #     run_sim_pair(list(range(5)), find_any_xfers_one_way_flow_pmms_top)
    # run_sim_pair(list(range(4)), one_way_flow_by_ratio_pmms)

    # def find_any_xfers_one_way_flow_pmms_top_with_orders(G, val_fns, goods):
    #     for order in [lambda x: val_fns[0][x]/val_fns[1][x],
    #                   lambda x: val_fns[1][x]/val_fns[0][x],
    #                   lambda x: val_fns[0][x],
    #                   lambda x: val_fns[1][x],
    #                   lambda x: -val_fns[0][x],
    #                   lambda x: -val_fns[1][x]]:
    #         a = find_any_xfers_one_way_flow_pmms_top(G, val_fns, goods, order)
    #         if a:
    #             return a
    #
    # run_exhaustive_sim_pair(list(range(5)), find_any_xfers_one_way_flow_pmms_top_with_orders)

    # random.seed(1)
    # for i in range(10000):
    #     run_sim_triplet(list(range(4)), yairs_algo)

    run_exhaustive_sim_triplet(list(range(5)), yairs_algo)
    # goods = list(range(5))
    # val_fns = {0: {0: 3, 1: 3, 2: 3, 3: 13, 4: 13},
    #            1: {0: 3, 1: 13, 2: 16, 3: 43, 4: 16},
    #            2: {0: 16, 1: 13, 2: 43, 3: 13, 4: 16}}
    #
    # agents = [0, 1, 2]
    #
    # G = nx.Graph()
    # G.add_nodes_from(agents)
    # G.add_edge(agents[0], agents[1])
    # G.add_edge(agents[1], agents[2])
    #
    # try:
    #     new_alloc = yairs_algo(G, val_fns, goods)
    # except Exception as e:
    #     print(val_fns)
    #     print(e)
    #
    # if not new_alloc:
    #     print(val_fns)
    #     print(str(yairs_algo) + " FAILED")
    #     sys.exit(0)