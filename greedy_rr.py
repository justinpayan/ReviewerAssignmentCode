# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import sys

from autoassigner import *
from copy import deepcopy
from utils import *


def get_greedy_rr(pra, covs, loads, best_revs):
    n = covs.shape[0]
    agent_selection_order = []

    # Maintain a list of agents in sorted order for round robin. At each iteration,
    # run round robin with all the remaining agents in the final slot at the end.
    # Add the agent with the best total marginal gain in usw from that round-robin run.
    marginal_gains_ub = {p: np.inf for p in range(n)}
    old_usw = 0
    global_max_usw = 0
    best_selection_order = None

    while len(agent_selection_order) < n:
        print(len(agent_selection_order))
        local_max_usw = -1
        next_agent = None
        global_max_usw_improved = False
        num_evaluated = 0
        for idx, a in enumerate(set(range(n)) - set(agent_selection_order)):
            if idx % 500 == 0:
                print("internal idx: ", idx)
            if marginal_gains_ub[a] > local_max_usw - old_usw:
                num_evaluated += 1
                a_rr_usw, _, _ = safe_rr_usw(agent_selection_order + [a], pra, covs, loads, best_revs)
                marginal_gains_ub[a] = a_rr_usw - old_usw
                if a_rr_usw > local_max_usw:
                    local_max_usw = a_rr_usw
                    next_agent = a
                if a_rr_usw > global_max_usw:
                    global_max_usw = a_rr_usw
                    global_max_usw_improved = True
        print(num_evaluated)
        print(global_max_usw)
        print(local_max_usw)
        print()

        agent_selection_order.append(next_agent)
        old_usw = local_max_usw

        if global_max_usw_improved:
            best_selection_order = deepcopy(agent_selection_order)

    return agent_selection_order, best_selection_order


def greedy_rr(pra, covs, loads, alloc_file):
    # Cut off the affinities so they're above 0. This is required for the objective to be submodular.
    pra[np.where(pra < 0)] = 0
    # pra /= np.max(pra)

    start = time.time()

    best_revs = np.argsort(-1 * pra, axis=0)

    complete_seln_order, _ = get_greedy_rr(pra, covs, loads, best_revs)
    alloc, _, _ = safe_rr(complete_seln_order, pra, covs, loads, best_revs)

    save_alloc(alloc, alloc_file)

    alg_time = time.time() - start

    print(alloc)
    print_stats(alloc, pra, covs, alg_time=alg_time)


def _greedy_rr_ordering(pra, covs, loads):
    # Cut off the affinities so they're above 0. This is required for the objective to be submodular.
    pra[np.where(pra < 0)] = 0
    # pra /= np.max(pra)

    best_revs = np.argsort(-1 * pra, axis=0)

    _, partial_selection_order = get_greedy_rr(pra, covs, loads, best_revs)
    return partial_selection_order


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    dataset = args.dataset

    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)

    if np.max(covs) != np.min(covs):
        print("covs must be homogenous")
        sys.exit(1)

    best_revs = np.argsort(-1 * scores, axis=0)

    complete_seln_order, partial_seln_order = get_greedy_rr(scores, covs, loads, best_revs)
    complete_alloc, _, _ = safe_rr(complete_seln_order, scores, covs, loads, best_revs)
    partial_alloc, _, _ = safe_rr(partial_seln_order, scores, covs, loads, best_revs)

    print(partial_alloc)

    with open("%s_greedy_init_order" % dataset, 'wb') as f:
        pickle.dump(partial_seln_order, f)
    # save_alloc(partial_alloc, args.alloc_file)
    save_alloc(complete_alloc, args.alloc_file)
    print_stats(complete_alloc, scores, covs)

    # with open("complete_order_cvpr_debug", "wb") as f:
    #     pickle.dump(complete_seln_order, f)
    # with open("partial_order_cvpr_debug", "wb") as f:
    #     pickle.dump(partial_seln_order, f)
    # save_alloc(complete_alloc, "complete_greedy_cvpr_debug")
    # save_alloc(partial_alloc, "partial_greedy_cvpr_debug")



