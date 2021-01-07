from itertools import chain, combinations, permutations, product
import numpy as np
import time
from eit_by_agents import *

"""Brute-force searching for a counter-example for the problem of finding near-equal (+1 is ok) point allocations
on a set of sets, which are potentially overlapping but no set is a subset of the other."""

# For each agent, compute the number of points of each group it's in.
# Then aggregate using the agg_fn, and pick the agent with the minimum of agg_fn over all groups.
# In case of ties, pick the agent with the lower number of individual points.
def agent_to_give(allocation, G, agg_fn):
    n = len(allocation)
    points_vec = compute_group_points(G, allocation)
    group_sizes = np.array([len(t) for t in G])

    weighted_points = points_vec / group_sizes

    agent_scores = []
    num_groups = []
    for a in range(n):
        idxs = get_group_containment(a, G)
        num_groups.append(idxs.size)
        a_score = agg_fn(weighted_points[idxs])
        agent_scores.append(a_score)

    # Sort and pick the worst-off agent
    sort_idxs = zip(agent_scores, allocation, range(n))
    # sort_idxs = zip(agent_scores, num_groups, range(n))
    sorted_by_agg_group_then_pts = sorted(sort_idxs)

    return sorted_by_agg_group_then_pts[0][2]


# Basically just picks an agent to give a point to repeatedly, according to the agent with the min of the agg_fn
# over all groups
def round_robin_solve(n, k, G, agg_fn):
    allocation = [1]*n
    choices = []
    while sum(allocation) < k:
        # Give to the agent with the smallest <agg_fn> wtd points over all groups containing it
        choices.append(agent_to_give(allocation, G, agg_fn))
        allocation[choices[-1]] += 1

    # while has_envious_pair(G, allocation, ef1b_prop) and len(choices):
    #     allocation[choices.pop()] -= 1

    while has_envious_pair(G, allocation, transfer_ef1) and len(choices):
        allocation[choices.pop()] -= 1

    # Exhaustively search all allocation completions... there may be one that happens to work
    # best_completion = []
    # best_util = sum(allocation)
    # for alloc_completion in product(range(n), repeat=k-sum(allocation)):
    #     for idx, a in enumerate(alloc_completion):
    #         allocation[a] += 1
    #         if sum(allocation) > best_util and not has_envious_pair(G, allocation, ef1b_prop):
    #             best_completion = alloc_completion[:idx+1]
    #             best_util = sum(allocation)
    #     for a in alloc_completion:
    #         allocation[a] -= 1
    #
    # for a in best_completion:
    #     allocation[a] += 1

    return allocation


def round_robin_min_solve(n, k, G):
    return round_robin_solve(n, k, G, np.min)


def round_robin_max_solve(n, k, G):
    return round_robin_solve(n, k, G, np.max)


def round_robin_avg_solve(n, k, G):
    return round_robin_solve(n, k, G, np.mean)


def bitfield(n, digits):
    bf = [1 if digit=='1' else 0 for digit in bin(n)[2:]]
    bf = [0]*(digits-len(bf)) + bf
    return bf


# Sample from the powerset of the powerset on n elements.
def sample_from_powerset_of_powerset(n):
    # to_return = []
    # which_sets = np.random.randint(2**(2**n))
    # print(which_sets)
    # which_sets = np.where(np.array(bitfield(which_sets)))[0].tolist()
    # print(which_sets)
    # for s in which_sets:
    #     to_return.append(np.where(np.array(bitfield(s)))[0].tolist())
    # return to_return
    to_return = []
    for i in range(5):
        elts = np.random.randint(2, 2**n)
        to_return.append(np.where(np.array(bitfield(elts, n)))[0].tolist())
    return to_return


# num_agents = 8
# alg_times = []
# brute_force_times = []
# for idx, set_of_sets in enumerate(powerset(powerset(range(num_agents)))):
#     if is_valid(num_agents, set_of_sets):
#         points = np.random.randint(num_agents*10, num_agents * 15)
#         problem_instance = (num_agents, points, set_of_sets)
#         # start = time.time()
#         # print(problem_instance)
#
#         alg_soln = round_robin_avg_solve(*problem_instance)
#         # alg_times.append(time.time()-start)
#
#         if alg_soln is None:
#             print(problem_instance)
#             # brute_soln = brute_force_solve(*problem_instance)
#             # print(problem_instance, None, None, brute_soln, sum(brute_soln))
#         elif sum(alg_soln) < points-3:
#             print(problem_instance, sum(alg_soln))
#             # brute_soln = brute_force_solve(*problem_instance)
#             # print(problem_instance, alg_soln, sum(alg_soln), brute_soln, sum(brute_soln))

def check_soln(soln, threshold, problem_instance):
    if soln is None or sum(soln) < threshold:
        # print(problem_instance, sum(soln))
        return False
    return True
        # brute_soln = brute_force_solve(*problem_instance)
        # print(problem_instance, alg_soln, sum(alg_soln), brute_soln, sum(brute_soln))

# num_agents = 20
# min_fail = 0
# max_fail = 0
# avg_fail = 0
# total = 0
# for idx in range(10000):
#     if idx % 100 == 0:
#         print(idx)
#     set_of_sets = sample_from_powerset_of_powerset(num_agents)
#     if is_valid(num_agents, set_of_sets):
#         points = np.random.randint(num_agents*10, num_agents * 15)
#         problem_instance = (num_agents, points, set_of_sets)
#         # start = time.time()
#         # print(problem_instance)
#
#         alg_soln_min = round_robin_min_solve(*problem_instance)
#         alg_soln_max = round_robin_max_solve(*problem_instance)
#         alg_soln_avg = round_robin_avg_solve(*problem_instance)
#         # alg_times.append(time.time()-start)
#
#         t = points - (num_agents - 3)
#         total += 1
#         min_fail += not int(check_soln(alg_soln_min, t, problem_instance))
#         max_fail += not int(check_soln(alg_soln_max, t, problem_instance))
#         avg_fail += not int(check_soln(alg_soln_avg, t, problem_instance))
#
#         if total % 10 == 0:
#             print(min_fail, max_fail, avg_fail, total)

if __name__ == '__main__':
    # num_agents = 7
    # total = 0
    # failures = 0
    # for idx, set_of_sets in enumerate(powerset(powerset(range(num_agents)))):
    #     if is_valid(num_agents, set_of_sets):
    #         # print(set_of_sets)
    #         for points in range(num_agents, num_agents*3):
    #             total += 1
    #             # points = np.random.randint(num_agents*2, num_agents * 3)
    #             problem_instance = (num_agents, points, set_of_sets)
    #             # print(problem_instance)
    #             # start = time.time()
    #             # alg_soln = minmax_eit_solve(*problem_instance)
    #             # alg_times.append(time.time()-start)
    #             # print(sum(alg_soln) == points)
    #             # start = time.time()
    #
    #             if total % 10000 == 0:
    #                 print(total, failures)
    #             alg_soln_avg = round_robin_avg_solve(*problem_instance)
    #             if sum(alg_soln_avg) != points:
    #                 failures += 1
    #                 print("FAIL")
    #                 print(problem_instance)

    problem_instance = (7, 20, ((0, 1), (0, 4), (0, 5), (2, 4, 5), (1, 3, 4, 5, 6)))

    print(brute_force_solve(*problem_instance), sum(brute_force_solve(*problem_instance)))
    # print(round_robin_max_solve(*problem_instance))
    # print(round_robin_min_solve(*problem_instance))
    print(round_robin_avg_solve(*problem_instance), sum(round_robin_avg_solve(*problem_instance)))