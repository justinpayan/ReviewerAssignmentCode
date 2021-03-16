import math
import numpy as np
from copy import deepcopy
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


def to_borda(val_fns):
    return np.argsort(np.argsort(val_fns, axis=1), axis=1) + 1


def calculate_conflicts(val_fns):
    conflicts = np.zeros(val_fns.shape)
    n, m = val_fns.shape
    for a in range(n):
        items_in_order = np.argsort(val_fns[a, :])
        # print(a)
        # print(items_in_order)
        for pref_num in range(m):
            # print(pref_num)
            # print(val_fns[:, items_in_order[pref_num]])
            # print(val_fns[:, items_in_order[pref_num]] >= (pref_num + 1))
            conflicts[a, m-pref_num-1] = np.sum(val_fns[:, items_in_order[pref_num]] >= (pref_num + 1)) - 1
            # print(conflicts[a, pref_num])
    # conflicts = np.sort(-1*conflicts, axis=1)*-1
    return conflicts


def optimal_rr(val_fns, num_rounds=1):

    n, m = val_fns.shape

    best_goods = np.argsort(-1*val_fns, axis=1)

    best_usw = 0
    worst_usw = np.inf

    for ordering in permutations(range(n)):
        remaining = np.ones((m))
        matrix_alloc = np.zeros((val_fns.shape), dtype=np.bool)

        for _ in range(num_rounds):
            for a in ordering:
                for r in best_goods[a, :]:
                    if remaining[r]:
                        remaining[r] -= 1
                        matrix_alloc[a, r] = 1
                        break
        usw = np.sum(matrix_alloc*val_fns)
        if usw > best_usw:
            best_usw = usw
        if usw < worst_usw:
            worst_usw = usw

    # print("in usw")
    # print(matrix_alloc)
    # print(val_fns)
    return best_usw, worst_usw


def rr_by_conflicts(conflicts, val_fns, num_rounds=1):
    matrix_alloc = np.zeros((val_fns.shape), dtype=np.bool)

    n, m = val_fns.shape

    remaining = np.ones((m))

    best_goods = np.argsort(-1*val_fns, axis=1)

    ordering = []
    for a in range(n):
        ordering.append((a, tuple(conflicts[a, :].tolist())))
    ordering = sorted(ordering, key=lambda x: x[1], reverse=True)
    ordering = [x[0] for x in ordering]

    for _ in range(num_rounds):
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


if __name__ == "__main__":
    # exhaustive_sim_triplet(efx_among_triplet)
    for s in range(100000):
        if s % 50 == 0:
            print(s)
        np.random.seed(s)

        n = 9
        m = 9
        valuations = np.random.rand(n, m)
        valuations = to_borda(valuations)
        # valuations = np.array([[3, 4, 1, 2],[2, 1, 3, 4], [1, 3, 2, 4], [2, 4, 1, 3]])

        # print(valuations)
        conflicts = calculate_conflicts(val_fns=valuations)

        opt, sub_opt = optimal_rr(valuations)

        if opt/rr_by_conflicts(conflicts, valuations) > 1:
            print(opt/rr_by_conflicts(conflicts, valuations))
            print(opt/sub_opt)
            print()
        # if not check_submodularity(valuations, rr_usw_no_steals):
        #     break
