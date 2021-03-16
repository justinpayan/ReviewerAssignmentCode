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


def rr_usw_no_steals(ordering, val_fns, num_rounds=1):
    if not len(ordering):
        return 0

    ordering = deepcopy(ordering)

    n, m = val_fns.shape

    matrix_alloc = np.zeros((val_fns.shape), dtype=np.bool)
    remaining = np.ones((m))

    best_goods = np.argsort(-1*val_fns, axis=1)

    def compute_assignment(ord, best_goods, remaining):
        rem = remaining.copy()
        round_alloc = np.zeros((val_fns.shape), dtype=np.bool)
        for a in ord:
            for r in best_goods[a, :]:
                if rem[r]:
                    rem[r] -= 1
                    round_alloc[a, r] = 1
                    break
        return round_alloc, rem

    # print("\nStarting")
    # print(ordering)

    for _ in range(num_rounds):
        round_alloc, remaining = compute_assignment(ordering, best_goods, remaining)
        matrix_alloc += round_alloc

    # print(matrix_alloc)

    remaining = np.ones((m))
    dirty_steal = True
    while dirty_steal:
        dirty_steal = False
        for a in ordering:
            # If it stole, then make the swap and redo

            # print("did {} steal?", a)
            # Run without that agent
            test_order = deepcopy(ordering)
            test_order.remove(a)
            # print(test_order)
            test_alloc, _ = compute_assignment(test_order, best_goods, remaining)

            # if np.sum(test_alloc * val_fns) <

            # See if it stole
            # for a_prime in test_order:
            #     # print("Did {} steal from {}?", a, a_prime)
            #     # If the value lost by a_prime is greater than the value gained by a, it's a steal
            #     value_gained_a = np.sum(matrix_alloc[a, :] * val_fns)
            #     value_lost_a_prime = np.sum(matrix_alloc[a_prime, :] * val_fns)
            #     value_gained_a_prime = np.sum(test_alloc[a_prime, :] * val_fns)
            #     if value_gained_a < value_gained_a_prime - value_lost_a_prime:
            #         dirty_steal = True
            #         # print("yes")
            #
            #         a_prime_idx = test_order.index(a_prime)
            #         test_order.insert(a_prime_idx + 1, a)
            #         ordering = test_order
            #         break
            #
            # if dirty_steal:
            #     break
    # print("final ordering: ", ordering)

    for _ in range(num_rounds):
        round_alloc, remaining = compute_assignment(ordering, best_goods, remaining)
        matrix_alloc += round_alloc

    # print("in usw")
    # print(matrix_alloc)
    # print(val_fns)
    return np.sum(matrix_alloc * val_fns)


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


def check_submodularity(val_fns, fun):
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
        # valuations = np.random.rand(n, m)
        # valuations = to_borda(valuations)
        valuations = np.array([[3, 4, 1, 2],[2, 1, 3, 4], [1, 3, 2, 4], [2, 4, 1, 3]])

        print(valuations)
        if not check_submodularity(valuations, rr_usw_no_steals):
            break
