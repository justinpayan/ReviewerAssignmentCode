# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import os
import random
import sys

from autoassigner import *
from copy import deepcopy
from utils import *


def run_probabilistic_max_exp(pra, covs, loads, best_revs):
    m, n = pra.shape

    num_comp = 5
    min_size = 1

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi", type=str, default="0")
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    phi = args.phi
    iter = args.iter

    pra = np.load("mallows_data/mallows_scores_{}_{}.npy".format(phi, iter))
    covs = np.load("mallows_data/mallows_covs_{}_{}.npy".format(phi, iter)).astype(np.int)
    loads = np.load("mallows_data/mallows_loads_{}_{}.npy".format(phi, iter)).astype(np.int64)

    best_revs = np.argsort(-1 * pra, axis=0)

    run_probabilistic_max_exp(pra, covs, loads, best_revs)




