from itertools import chain, combinations, permutations
import numpy as np
import time

"""Brute-force searching for a counter-example for the problem of finding near-equal (+1 is ok) point allocations
on a set of sets, which are potentially overlapping but no set is a subset of the other."""

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(0, len(s)+1))

# from http://jeromekelleher.net/generating-integer-partitions.html
# Generates all partitions of n
def accel_asc(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

# Pass in the number of elements we have n, and pass in the number of points to allocate k
# Return all permutations of all partitions of k into n values, including allocating 0 to some elements.
# This is basically all the ways we could divvy up k points among n elements.
def generate_all_point_allocations(n, k):
    for partition in accel_asc(k):
        if len(partition) <= n:
            while len(partition) < n:
                partition.append(0)
            for perm in set(permutations(partition)):
                yield perm


def ef1(pair, allocation):
    alloc_pair0 = sum([allocation[i] for i in pair[0]])
    alloc_pair1 = sum([allocation[i] for i in pair[1]])
    return abs(alloc_pair0 - alloc_pair1) <= 1


# envy is only computed for parts of the sets that dont overlap
def ef1_nonoverlapping(pair, allocation):
    p0 = set(pair[0]) - set(pair[1])
    p1 = set(pair[1]) - set(pair[0])
    if len(p0) == 0 or len(p1) == 0:
        return True
    alloc_pair0 = sum([allocation[i] for i in p0])
    alloc_pair1 = sum([allocation[i] for i in p1])
    return abs(alloc_pair0 - alloc_pair1) <= 1


# EF1A holds if we cannot multiply by ratio of group sizes, then remove an item, and still prefer the other bundle
def ef1a_prop(pair, allocation):
    alloc_pair0 = sum([allocation[i] for i in pair[0]])
    alloc_pair1 = sum([allocation[i] for i in pair[1]])
    if ((alloc_pair1 * float(len(pair[0])) / len(pair[1]))-1 > alloc_pair0) or \
       ((alloc_pair0 * float(len(pair[1])) / len(pair[0]))-1 > alloc_pair1):
        return False
    else:
        return True

def ef1a_prop_nonoverlapping(pair, allocation):
    p0 = set(pair[0]) - set(pair[1])
    p1 = set(pair[1]) - set(pair[0])
    if len(p0) == 0 or len(p1) == 0:
        return True
    alloc_pair0 = sum([allocation[i] for i in p0])
    alloc_pair1 = sum([allocation[i] for i in p1])
    if ((alloc_pair1 * float(len(p0)) / len(p1))-1 > alloc_pair0) or \
       ((alloc_pair0 * float(len(p1)) / len(p0))-1 > alloc_pair1):
        return False
    else:
        return True


# EF1B holds if we cannot remove an item from the other group then multiply by ratio of group sizes and still prefer
# the other bundle. This is equivalent to WEF1 from Chakraborty et al.
def ef1b_prop(pair, allocation):
    alloc_pair0 = sum([allocation[i] for i in pair[0]])
    alloc_pair1 = sum([allocation[i] for i in pair[1]])
    if ((alloc_pair1-1) * float(len(pair[0]))/len(pair[1]) > alloc_pair0) or \
       ((alloc_pair0-1) * float(len(pair[1])) / len(pair[0]) > alloc_pair1):
        return False
    else:
        return True


# WWEF1 from Chakraborty et al.
def wwef1(pair, allocation):
    alloc_pair0 = sum([allocation[i] for i in pair[0]])
    alloc_pair1 = sum([allocation[i] for i in pair[1]])
    if ((alloc_pair0 * float(len(pair[1])) / len(pair[0]) >= alloc_pair1 - 1) or
        ((alloc_pair0+1) * float(len(pair[1])) / len(pair[0]) >= alloc_pair1)) and \
       ((alloc_pair1 * float(len(pair[0])) / len(pair[1]) >= alloc_pair0 - 1) or
        ((alloc_pair1+1) * float(len(pair[0])) / len(pair[1]) >= alloc_pair0)):
        return True
    else:
        return False


# Transfer EF1, mentioned in Chakraborty et al. but comes from earlier paper I think?
def transfer_ef1(pair, allocation):
    alloc_pair0 = sum([allocation[i] for i in pair[0]])
    alloc_pair1 = sum([allocation[i] for i in pair[1]])
    if ((alloc_pair0+1) * float(len(pair[1])) / len(pair[0]) >= alloc_pair1 - 1) and \
       ((alloc_pair1+1) * float(len(pair[0])) / len(pair[1]) >= alloc_pair0 - 1):
        return True
    else:
        return False


def ef1b_prop_nonoverlapping(pair, allocation):
    p0 = set(pair[0]) - set(pair[1])
    p1 = set(pair[1]) - set(pair[0])
    if len(p0) == 0 or len(p1) == 0:
        return True
    alloc_pair0 = sum([allocation[i] for i in p0])
    alloc_pair1 = sum([allocation[i] for i in p1])
    if ((alloc_pair1-1) * float(len(p0))/len(p1) > alloc_pair0) or \
       ((alloc_pair0-1) * float(len(p1)) / len(p0) > alloc_pair1):
        return False
    else:
        return True


def has_envious_pair(groups, allocation, condition):
    for pair in combinations(groups, 2):
        if not condition(pair, allocation):
            print(pair)
            return True
    return False


# Find an allocation, among the pregenerated list, that is 'valid' for this set of sets
# A 'valid' allocation meets the condition function. Condition might be that EF1 holds,
# or it might be that EF1 holds up to proportionality.
def find_allocation(groups, all_allocations, condition):
    # allocation is the amount of points allocated to each individual
    # print("find_allocation")
    # print(groups)
    # print(amt)
    # print(all_allocations)
    for allocation in all_allocations:
        allocation_ok = True
        for idx in range(len(allocation)):
            if allocation[idx] > 0:
                found_idx = False
                for g in groups:
                    if idx in g:
                        found_idx = True
                if not found_idx:
                    allocation_ok = False

        # print(allocation, allocation_ok)
        if allocation_ok:
            allocation_ok = not has_envious_pair(groups, allocation, condition)

        if allocation_ok:
            return allocation
    return None


# Every agent must have at least one group, and also no group can be a strict subset of another.
def is_valid(num_agents, set_of_sets):
    if len(set_of_sets) == 1:
        return False
    for i in range(num_agents):
        found_agent = False
        found_set_without_agent = False
        for t in set_of_sets:
            if i in t:
                found_agent = True
            else:
                found_set_without_agent = True
            if found_agent and found_set_without_agent:
                break
        if not (found_agent and found_set_without_agent):
            return False

    for pair in combinations(set_of_sets, 2):
        if set(pair[0]).issubset(set(pair[1])):
            return False
    return True


# Takes in a number of agents, a maximum point value k, and a set of groups of agents G,
# and determines the maximum number of points k' <= k that we can allocate to agents 1-n
# so that there's no weighted envy up to more than 1 item for pairs of groups in G.
def brute_force_solve(n, k, G, criterion):
    k_prime = k
    while k_prime > 0:
        all_allocations = list(generate_all_point_allocations(n, k_prime))
        soln = find_allocation(G, all_allocations, criterion)
        if soln:
            return soln
        else:
            k_prime -= 1
    return None


# Takes in a number of agents, a maximum point value k, and a set of groups of agents G,
# and determines the maximum number of points k' <= k that we can allocate to agents 1-n
# so that there's no weak weighted envy up to more than 1 item for pairs of groups in G.
def brute_force_solve_wwef1(n, k, G):
    return brute_force_solve(n, k, G, wwef1)


def brute_force_solve_transfer_ef1(n, k, G):
    return brute_force_solve(n, k, G, transfer_ef1)


def compute_group_points(G, allocation):
    points_vec = np.zeros(len(G))
    for idx, t in enumerate(G):
        for agent in t:
            points_vec[idx] += allocation[agent]
    return points_vec


def get_group_containment(a, G):
    containing_groups = []
    for idx, t in enumerate(G):
        if a in t:
            containing_groups.append(idx)
    return np.array(containing_groups)


# For transfer TO, we want the agent with minimum maximum envy. If there isn't a unique agent, pick the
# one with the lowest points individually
def get_transfer_to(max_envies, allocation):
    sort_idxs = zip(max_envies, allocation, range(len(allocation)))
    sorted_by_envy_then_pts = sorted(sort_idxs)
    return [t[2] for t in sorted_by_envy_then_pts]


# Same for the agent we're transferring FROM
def get_transfer_from(min_envies, allocation):
    sort_idxs = zip(min_envies, allocation, range(len(allocation)))
    sorted_by_envy_then_pts = sorted(sort_idxs, reverse=True)
    return [t[2] for t in sorted_by_envy_then_pts]


def get_next_allocation(transfer_to, transfer_from, allocation, allocations_tried):
    n = len(allocation)
    for from_idx in range(n):
        if allocation[transfer_from[from_idx]] > 0:
            allocation[transfer_from[from_idx]] -= 1
            for to_idx in range(n):
                # Transfer a point
                allocation[transfer_to[to_idx]] += 1

                # If we haven't seen this before, we are happy. If we have, back up and try another.
                if str(allocation) in allocations_tried:
                    allocation[transfer_to[to_idx]] -= 1
                else:
                    return allocation
            # Back up and try transferring FROM a different agent
            allocation[transfer_from[from_idx]] += 1
    return None


def minmax_eit_solve(n, k, G):
    # print(n, k, G)
    to_give_to_all = k//n
    allocation = [to_give_to_all]*n
    for i in range(k-sum(allocation)):
        allocation[i] += 1

    # Keep track of the allocations we've already seen. If we repeat one, drop a point.
    allocations_tried = set()
    allocations_tried.add(str(allocation))

    while sum(allocation) > 0 and has_envious_pair(G, allocation, ef1b_prop):
        points_vec = compute_group_points(G, allocation)
        group_sizes = np.array([len(t) for t in G])

        # envy_ij is the envy that group i feels for group j... where envy is computed by the
        # improvement of taking group j's points and then rescaling, over group i's original point value
        group_size_ratios = np.outer(group_sizes, 1/group_sizes)
        # envied_group_points_wtd = group_size_ratios * points_vec.reshape((1, -1))
        # envy = envied_group_points_wtd - points_vec.reshape((-1, 1))
        # Let's try computing envy up to 1 item instead...
        envied_group_points_wtd = group_size_ratios * (points_vec.reshape((1, -1))-1)
        envy = envied_group_points_wtd - points_vec.reshape((-1, 1))

        # Find the agent with the minimum maximum envy. We transfer TO that agent
        # Find the agent with the maximum minimum envy. We transfer FROM that agent
        max_envies = []
        min_envies = []
        for a in range(n):
            idxs = get_group_containment(a, G)
            envy_to_groups_containing_agent = np.delete(envy[:, idxs], idxs, axis=0)
            max_envies.append(np.max(envy_to_groups_containing_agent))
            min_envies.append(np.min(envy_to_groups_containing_agent))

        transfer_from = get_transfer_from(min_envies, allocation)
        transfer_to = get_transfer_to(max_envies, allocation)

        allocation = get_next_allocation(transfer_to, transfer_from, allocation, allocations_tried)

        if allocation:
            allocations_tried.add(str(allocation))
            # if len(allocations_tried) % 10000 == 0:
            #     print(len(allocations_tried))
        else:
            return minmax_eit_solve(n, k-1, G)

    return allocation




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


if __name__ == '__main__':
    # problem_instance = (6, 33, ((0, 1), (0, 2), (0, 3, 4, 5), (1, 2, 3, 4, 5)))
    # (6, 32, ((0, 1), (0, 4), (0, 2, 3, 5), (1, 2, 3, 4, 5)))
    for points in range(7, 100):
        problem_instance = (9, points, ((0,1,2,3,4), (0,5), (1,5,6), (2,6,7), (3,7,8), (4,8)), ef1b_prop_nonoverlapping)
        print(points)
        print(sum(brute_force_solve(*problem_instance)) == points)

    print(brute_force_solve(*problem_instance))
    # print(minmax_eit_solve(*problem_instance))

    num_agents = 5
    total = 0
    for idx, set_of_sets in enumerate(powerset(powerset(range(num_agents)))):
        if is_valid(num_agents, set_of_sets):
            # print(set_of_sets)
            for points in range(num_agents, num_agents*2):
                total += 1
                # points = np.random.randint(num_agents*2, num_agents * 3)
                problem_instance = (num_agents, points, set_of_sets, ef1b_prop_nonoverlapping)
                # print(problem_instance)
                # start = time.time()
                # alg_soln = minmax_eit_solve(*problem_instance)
                # alg_times.append(time.time()-start)
                # print(sum(alg_soln) == points)
                # start = time.time()
                brute_soln = brute_force_solve(*problem_instance)
                # brute_force_times.append(time.time()-start)

                if sum(brute_soln) != points:
                    print(problem_instance)
                    print("FAIL")
                if total % 10000 == 0:
                    print(total)
            # if idx % 10 == 0:
            #     print("Avg time alg: %.4f\nAvg time brute force: %.4f\n" %
            #           (float(np.mean(alg_times)), float(np.mean(brute_force_times))))
            # if sum(alg_soln) != sum(brute_soln):
            #     print(num_agents, points, set_of_sets, alg_soln, sum(alg_soln), brute_soln, sum(brute_soln))

            # if alg_soln is None:
            #     print(problem_instance)
            #     brute_soln = brute_force_solve(*problem_instance)
            #     print(problem_instance, None, None, brute_soln, sum(brute_soln))
            # elif sum(alg_soln) < points:
            #     print(problem_instance)
            #     brute_soln = brute_force_solve(*problem_instance)
            #     print(problem_instance, alg_soln, sum(alg_soln), brute_soln, sum(brute_soln))

    # num_agents = 6
    # failures = 0
    # total = 0
    # for idx in range(10000):
    #     set_of_sets = sample_from_powerset_of_powerset(num_agents)
    #     if is_valid(num_agents, set_of_sets):
    #         # print(set_of_sets)
    #         points = np.random.randint(num_agents*5, num_agents * 10)
    #         problem_instance = (num_agents, points, set_of_sets)
    #         # print(problem_instance)
    #         # start = time.time()
    #         brute_soln = brute_force_solve(*problem_instance, ef1b_prop_nonoverlapping)
    #         if sum(brute_soln) == points:
    #             total += 1
    #         else:
    #             total += 1
    #             failures += 1
    #             # total += 1
    #             # alg_soln = minmax_eit_solve(*problem_instance)
    #             # if sum(alg_soln) != points:
    #             #     failures += 1
    #             #     print("failures %d of %d" % (failures, total))
    #             #     print(problem_instance)
    #
    #     if idx % 10 == 0:
    #         print(idx, total, failures)
    #         # alg_times.append(time.time()-start)
    #         # print(sum(alg_soln) == points)
    #         # start = time.time()
    #         # brute_soln = brute_force_solve(*problem_instance)
    #         # brute_force_times.append(time.time()-start)
    #
    #         # if idx % 10 == 0:
    #         #     print("Avg time alg: %.4f\nAvg time brute force: %.4f\n" %
    #         #           (float(np.mean(alg_times)), float(np.mean(brute_force_times))))
    #         # if sum(alg_soln) != sum(brute_soln):
    #         #     print(num_agents, points, set_of_sets, alg_soln, sum(alg_soln), brute_soln, sum(brute_soln))
    #
    #         # if alg_soln is None:
    #         #     print(problem_instance)
    #         #     brute_soln = brute_force_solve(*problem_instance)
    #         #     print(problem_instance, None, None, brute_soln, sum(brute_soln))
    #         # elif sum(alg_soln) < points:
    #         #     print(problem_instance)
    #         #     brute_soln = brute_force_solve(*problem_instance)
    #         #     print(problem_instance, alg_soln, sum(alg_soln), brute_soln, sum(brute_soln))
