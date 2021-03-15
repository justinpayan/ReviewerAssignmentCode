import math
import numpy as np
from itertools import permutations, chain, combinations, product


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


def rr_usw(ordering, val_fns, allocate_all=True):
    if not len(ordering):
        return 0

    matrix_alloc = np.zeros((val_fns.shape), dtype=np.bool)

    n, m = val_fns.shape

    remaining = np.ones((m))

    best_goods = np.argsort(-1*val_fns, axis=1)

    if not allocate_all:
        for _ in range(math.floor(m / n)):
            for a in ordering:
                for r in best_goods[a, :]:
                    if remaining[r]:
                        remaining[r] -= 1
                        matrix_alloc[a, r] = 1
                        break
    else:
        while np.sum(remaining) > 0:
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


def check_submodularity(val_fns):
    n, m = val_fns.shape

    # For all proper subsets of agents
    for a_set in powerset(range(n)):
        a_set = set(a_set)
        if a_set != set(range(n)) and len(a_set):
            for subset in powerset(a_set):
                subset = set(subset)
                if subset != a_set:
                    for e in set(range(n)) - a_set:
                        for order in permutations(a_set):
                            order = list(order)
                            for i in range(len(a_set)):
                                # print("calling usw calls")
                                suborder = [j for j in order if j in subset]
                                # print(order)
                                # print(suborder)
                                # print(i)
                                # print(e)
                                usw_order = rr_usw(order, val_fns)
                                usw_suborder = rr_usw(suborder, val_fns)
                                order.insert(i, e)
                                suborder = [j for j in order if j in (subset | {e})]
                                usw_order_e = rr_usw(order, val_fns)
                                usw_suborder_e = rr_usw(suborder, val_fns)
                                if usw_order_e - usw_order > usw_suborder_e - usw_suborder:
                                    print(usw_order_e)
                                    print(usw_suborder_e)
                                    print(usw_order)
                                    print(usw_suborder)
                                    print(order, suborder, e, i)
                                    return False
    return True


def to_borda(val_fns):
    return np.argsort(np.argsort(val_fns, axis=1), axis=1) + 1


if __name__ == "__main__":
    # exhaustive_sim_triplet(efx_among_triplet)
    for s in range(100000):
        if s % 50 == 0:
            print(s)
        np.random.seed(s)

        n = 4
        m = 4
        valuations = np.random.rand(n, m)
        valuations = to_borda(valuations)

        print(valuations)
        if not check_submodularity(valuations):
            break
