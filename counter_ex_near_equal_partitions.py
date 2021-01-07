from itertools import chain, combinations, permutations

"""Brute-force searching for a counter-example for the problem of finding near-equal (+1 is ok) point allocations
on a set of sets, which are potentially overlapping but no set is a subset of the other."""

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

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
# the other bundle.
def ef1b_prop(pair, allocation):
    alloc_pair0 = sum([allocation[i] for i in pair[0]])
    alloc_pair1 = sum([allocation[i] for i in pair[1]])
    if ((alloc_pair1-1) * float(len(pair[0]))/len(pair[1]) > alloc_pair0) or \
       ((alloc_pair0-1) * float(len(pair[1])) / len(pair[0]) > alloc_pair1):
        return False
    else:
        return True


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
            for pair in combinations(groups, 2):
                # print(pair)
                if not condition(pair, allocation):
                    allocation_ok = False

        if allocation_ok:
            print("found allocation")
            print(allocation)
            print()
            return True
    print("No allocation found")
    print()
    return False


def is_valid(set_of_sets):
    if len(set_of_sets) == 1:
        return False
    for pair in combinations(set_of_sets, 2):
        if set(pair[0]).issubset(set(pair[1])):
            return False
    return True


# This is only for a very specific set of groups, where we have the n choose n-1 groups and then
# another single group with n elements.
# Let's also only consider an even n for now. n is num_agents, btw.
def generate_point_allocations_up_to_isomorphism(num_agents, points):
    return None
    for partition in accel_asc(k):
        if len(partition) <= n:
            while len(partition) < n:
                partition.append(0)
            # TODO: Frankly, I'm not sure what to swap the below two lines with, since we
            # TODO:
            # for perm in set(permutations(partition)):
            #     yield perm


for points in range(63,15,-1):
    print(points)
    num_agents = 8
    all_allocations = generate_all_point_allocations(num_agents, points)
    set_of_sets = [(0,1,2), (1,2,3), (0,2,3), (0,1,3), (4,5,6,7)]
    print(is_valid(set_of_sets))
    print(find_allocation(set_of_sets, all_allocations, ef1b_prop))
    print()


# num_agents = 6
# points = 21
# all_allocations = list(generate_all_point_allocations(num_agents, points))
# idx = 0
# for set_of_sets in powerset(powerset(range(num_agents))):
#     if idx % 1e2 == 0:
#         print(idx)
#         print(float(idx)/(2**(2**num_agents)))
#         print(flush=True)
#     if is_valid(set_of_sets):
#         if not find_allocation(set_of_sets, all_allocations, ef1b_prop_nonoverlapping):
#             print(set_of_sets)
#     idx += 1


