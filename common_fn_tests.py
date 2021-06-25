import os
import random
import sys

from autoassigner import *
from copy import deepcopy
from utils import *


def run_probabilistic_max_exp(pra, covs, loads, best_revs, min_size=1, num_comp=5):
    m, n = pra.shape

    wins = []
    correct = []
    set_sizes = []
    num_deletes = []

    for i in range(1000):
        if i % 10 == 0 and i > 0:
            print(i)
            print(wins)
            print(correct)
            print(set_sizes)
            print(num_deletes)
            wc = list(zip(wins, correct))
            for i in range(num_comp, num_comp-(num_comp+1)//2, -1):
                # print(i)
                # print(wc)
                # print([1 - x[1] for x in wc if x[0] >= i])
                num_incorrect = sum([1-x[1] for x in wc if x[0] >= i])
                print("Num incorrect when wins >= {}: {}".format(i, num_incorrect))
            # print("num incorrect: ", len(correct) - sum(correct))
            print(flush=True)

        num_agents_X = np.random.randint(min_size, n-1)
        # num_agents_X_e1_e2 = i % (n-3) + 3
        agents_X = set(random.sample(range(n), num_agents_X))

        positions_X = sorted(range(n), key=lambda x: random.random())[:num_agents_X]
        position_map = dict(zip(sorted(agents_X), positions_X))

        X = {(a, position_map[a]) for a in agents_X}

        # e = random.sample(set(product(range(n), range(n))) - X, 1)[0]
        open_pos = set(range(n)) - set(positions_X)
        pool = set(product(agents_X, open_pos))
        open_ag = set(range(n)) - agents_X
        pool |= set(product(open_ag, positions_X))
        e = random.sample(pool, 1)[0]

        del_1 = set()
        del_2 = set()

        if e[0] in agents_X or e[1] in positions_X:
            for f in X:
                if f[0] == e[0]:
                    del_1.add(f)
                elif f[1] == e[1]:
                    del_2.add(f)
        X_prime = X - (del_1 | del_2)
        X_prime.add(e)

        # Run num_comp completions of both, record number of times X beats X_prime
        winners = []
        for _ in range(num_comp):
            val_X = estimate_expected_safe_rr_usw(X, pra, covs, loads, best_revs, n_iters=1)
            val_X_prime = estimate_expected_safe_rr_usw(X_prime, pra, covs, loads, best_revs, n_iters=1)
            winners.append(val_X > val_X_prime)
        X_wins = int(np.sum(np.array(winners)))
        winner = X_wins >= num_comp/2  # X won more that X_prime
        num_wins = max(X_wins, num_comp - X_wins)

        # compute the true max of both
        max_X = max_safe_rr_usw(X, pra, covs, loads, best_revs)
        max_X_prime = max_safe_rr_usw(X_prime, pra, covs, loads, best_revs)
        true_winner = max_X >= max_X_prime

        wins.append(num_wins)
        correct.append(winner == true_winner or np.isclose(max_X, max_X_prime))
        set_sizes.append((len(X), len(X_prime)))
        num_deletes.append(len(del_1) + len(del_2))


def run_submod_exp(pra, covs, loads, best_revs, norm=None, min_size=5):
    m, n = pra.shape

    violator_sizes = []
    violation_amounts = []
    gamma_values = []
    non_monotone = 0

    for i in range(10000):
        if i % 1 == 0 and i > 0:
            print(i)
            print(len(violator_sizes))
            # print(violator_sizes)
            # print(violation_amounts)
            # print(sorted(gamma_values))
            if gamma_values:
                print(min(gamma_values))
            print(non_monotone)

        num_agents_Ye = np.random.randint(min_size, n)
        _agents_Ye = list(random.sample(range(n), num_agents_Ye))
        bin_agents_Ye = np.zeros(n)
        bin_agents_Ye[_agents_Ye] = 1
        e = np.random.choice(np.where(bin_agents_Ye)[0])
        bin_agents_Y = bin_agents_Ye.copy()
        bin_agents_Y[e] = 0
        agents_Ye = np.where(bin_agents_Ye)[0].tolist()
        agents_Y = np.where(bin_agents_Y)[0].tolist()

        num_agents_X = np.random.randint(min_size-1, num_agents_Ye)
        _agents_X = list(random.sample(agents_Y, num_agents_X))
        bin_agents_X = np.zeros(n)
        bin_agents_X[_agents_X] = 1
        bin_agents_Xe = bin_agents_X.copy()
        bin_agents_Xe[e] = 1
        agents_Xe = np.where(bin_agents_Xe)[0].tolist()
        agents_X = np.where(bin_agents_X)[0].tolist()

        positions_Ye = sorted(range(n), key=lambda x: random.random())[:num_agents_Ye]
        position_map = dict(zip(sorted(agents_Ye), positions_Ye))

        X = {(a, position_map[a]) for a in agents_X}
        Xe = {(a, position_map[a]) for a in agents_Xe}
        Y = {(a, position_map[a]) for a in agents_Y}
        Ye = {(a, position_map[a]) for a in agents_Ye}

        # total_order = sorted(agents_Ye, key=lambda x: random.random())
        # order_map = {i: idx for idx, i in enumerate(total_order)}
        #
        # agents_Ye = sorted(agents_Ye, key=lambda x: order_map[x])
        # agents_Y = sorted(agents_Y, key=lambda x: order_map[x])
        # agents_Xe = sorted(agents_Xe, key=lambda x: order_map[x])
        # agents_X = sorted(agents_X, key=lambda x: order_map[x])

        # num_agents_X_e1_e2 = np.random.randint(min_size, n)
        # agents_X_e1_e2 = set(random.sample(range(n), num_agents_X_e1_e2))
        # e1 = random.sample(agents_X_e1_e2, 1)[0]
        # e2 = random.sample(agents_X_e1_e2 - {e1}, 1)[0]
        # agents_X = agents_X_e1_e2 - {e1, e2}
        #
        # positions_X_e1_e2 = sorted(range(n), key=lambda x: random.random())[:num_agents_X_e1_e2]
        # position_map = dict(zip(sorted(agents_X_e1_e2), positions_X_e1_e2))
        #
        # X = {(a, position_map[a]) for a in agents_X}
        # X_e1 = {(a, position_map[a]) for a in agents_X | {e1}}
        # X_e2 = {(a, position_map[a]) for a in agents_X | {e2}}
        # X_e1_e2 = {(a, position_map[a]) for a in agents_X | {e1, e2}}

        vals = []

        # for inp in [X, X_e1, X_e2, X_e1_e2]:
        for inp in [X, Y, Xe, Ye]:
            seln_order = [x[0] for x in sorted(inp, key=lambda x: x[1])]
            # vals.append(max_safe_rr_usw(inp, pra, covs, loads, best_revs, norm))
            # vals.append(estimate_expected_safe_rr_usw(inp, pra, covs, loads, best_revs, n_iters=2, normalizer=norm))
            vals.append(safe_rr_usw(seln_order, pra, covs, loads, best_revs)[0]*len(seln_order)**1.1)

        diff_lhs = vals[2] - vals[0]
        diff_rhs = vals[3] - vals[1]

        if diff_lhs < diff_rhs and not np.isclose(diff_lhs, diff_rhs):
            violator_sizes.append(num_agents_Ye)
            violation_amounts.append(diff_rhs - diff_lhs)
            gamma_values.append(diff_lhs/diff_rhs)
        if diff_lhs < 0 or diff_rhs < 0:
            print(len(X))
            print(len(Y))
            print(len(Xe))
            print(len(Ye))
            print(vals)
            non_monotone += 1