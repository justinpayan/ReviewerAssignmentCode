# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import os
import sys

from autoassigner import *
from copy import deepcopy
from utils import *


# Return the usw of running round robin on the agents in the list "seln_order"
def rr_usw(seln_order, pra, covs, loads, best_revs):
    # rr_alloc, rev_loads_remaining, matrix_alloc = rr(seln_order, pra, covs, loads, best_revs)
    _, rev_loads_remaining, matrix_alloc = rr(seln_order, pra, covs, loads, best_revs, output_alloc=False)
    # _usw = usw(rr_alloc, pra)
    _usw = np.sum(matrix_alloc * pra)
    # print("USW ", time.time() - start)
    return _usw, rev_loads_remaining, matrix_alloc


def rr(seln_order, pra, covs, loads, best_revs, output_alloc=True):
    if output_alloc:
        alloc = {p: list() for p in seln_order}
    matrix_alloc = np.zeros((pra.shape), dtype=np.bool)

    loads_copy = loads.copy()

    # Assume all covs are the same
    if output_alloc:
        for _ in range(covs[seln_order[0]]):
            for a in seln_order:
                for r in best_revs[:, a]:
                    if loads_copy[r] > 0 and r not in alloc[a]:
                        loads_copy[r] -= 1
                        alloc[a].append(r)
                        matrix_alloc[r, a] = 1
                        break
        return alloc, loads_copy, matrix_alloc
        # best_revs = np.argmax(pra[:, seln_order] * (matrix_alloc[:, seln_order] == 0) * (loads_copy > 0).reshape((-1,1)), axis=0)

        # for idx, a in enumerate(seln_order):
        #     # Allocate a reviewer which still has slots and hasn't already been allocated to this paper
        #     if loads_copy[best_revs[idx]] > 0:
        #         loads_copy[best_revs[idx]] -= 1
        #         alloc[a].append(best_revs[idx])
        #         matrix_alloc[best_revs[idx], a] = 1
        #     else:
        #         # Need to recompute, since this reviewer has no slots left
        #         best_rev = np.argmax(pra[:, a] * (loads_copy > 0) * (matrix_alloc[:, a] == 0))
        #         loads_copy[best_rev] -= 1
        #         alloc[a].append(best_rev)
        #         matrix_alloc[best_rev, a] = 1

        # This was the original way I was doing round-robin. I think the way above is at least a little bit faster.
        # for a in seln_order:
        #     # Allocate a reviewer which still has slots and hasn't already been allocated to this paper
        #     best_rev = np.argmax(pra[:, a] * (loads_copy > 0) * (matrix_alloc[:, a] == 0))
        #     loads_copy[best_rev] -= 1
        #     alloc[a].append(best_rev)
        #     matrix_alloc[best_rev, a] = 1
    else:
        for _ in range(covs[seln_order[0]]):
            for a in seln_order:
                for r in best_revs[:, a]:
                    if loads_copy[r] > 0 and matrix_alloc[r, a] == 0:
                        loads_copy[r] -= 1
                        matrix_alloc[r, a] = 1
                        break
        # return is wayyy below


        # THE CODE FROM HERE TO "END" WAS AN ATTEMPT TO VECTORIZE RR WHICH DID NOT SPEED IT UP
        # This tells each paper which idx of best_rev they will take from next
        # print("starting RR")
        # n = covs.shape[0]
        # which_choice = np.zeros(n, dtype=np.int)
        # for _ in range(covs[seln_order[0]]):
        #     # which_revs = best_revs[which_choice, :][0, :]
        #     which_revs = best_revs[which_choice, range(n)]
        #
        #     # For each reviewer which exceeded their load, find the agents which need to change
        #     values, counts = np.unique(which_revs, return_counts=True)
        #     # print(which_revs)
        #     # print(counts > loads_copy[values])
        #     while np.any(counts > loads_copy[values]):
        #         # These are the reviewers whose loads were exceeded this round, and we take the first.
        #         # I think this code does all the reviewers with violating loads at once.
        #         # ind = np.where(counts > loads_copy[values])[0][0]
        #         # v = values[ind]
        #         inds = np.where(counts > loads_copy[values])[0]
        #         violated_reviewers = values[inds]
        #         which_agents_to_change = loads_copy[violated_reviewers]
        #
        #         # This will give us all agents which have a violated reviewer, but how do we suppress
        #         # the incrementation of the first few agents (who can keep that reviewer).
        #         num_violated_revs = violated_reviewers.shape[0]
        #         violations = np.where(np.tile(which_revs, (num_violated_revs, 1))
        #                               == violated_reviewers.reshape((-1, 1)))
        #
        #         agents_to_update = np.zeros((num_violated_revs, n), dtype=np.int)
        #         agents_to_update[violations[0], violations[1]] = 1
        #
        #         # TODO: we need to set the first which_agents_to_change nonzero elements of each row to 0
        #         # TODO: the code commented below does not slice correctly, and also it gets the first i
        #         # TODO: elements instead of the first i nonzero elements
        #         # agents_to_update[range(num_violated_revs), :which_agents_to_change] = 0
        #         cs = np.cumsum(agents_to_update, axis=1)
        #         mask = (cs > which_agents_to_change.reshape((-1, 1)))
        #         agents_to_update *= mask
        #         # for i in which_agents_to_change:
        #
        #         agents_to_update = np.sum(agents_to_update, axis=0)
        #         which_choice[np.where(agents_to_update)] += 1
        #
        #         which_revs = best_revs[which_choice, range(n)]
        #         values, counts = np.unique(which_revs, return_counts=True)
        #
        #         # for idx in range(violated_reviewers.shape[0]):
        #         #     violating_agents = np.where(which_revs == violated_reviewers[idx])[0][which_agents_to_change[idx]:]
        #         #     which_choice[violating_agents] += 1
        #         # # which_revs = best_revs[which_choice, :][0, :]
        #         # which_revs = best_revs[which_choice, range(n)]
        #
        #         # violating_agents = np.where(which_revs in vals)
        #
        #         # violating_agents = np.where(which_revs == v)[0][loads_copy[v]:]
        #         # which_choice[violating_agents] += 1
        #         # which_revs = best_revs[which_choice, :]
        #
        #     matrix_alloc[which_revs, :] = 1
        #     values, counts = np.unique(which_revs, return_counts=True)
        #     loads_copy[values] -= counts
        #
        #     which_choice += 1
        # END

        return None, loads_copy, matrix_alloc


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
        for a in set(range(n)) - set(agent_selection_order):
            if marginal_gains_ub[a] > local_max_usw - old_usw:
                num_evaluated += 1
                a_rr_usw, _, _ = rr_usw(agent_selection_order + [a], pra, covs, loads, best_revs)
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
    alloc, _, _ = rr(complete_seln_order, pra, covs, loads, best_revs)

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

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)

    if np.max(covs) != np.min(covs):
        print("covs must be homogenous")
        sys.exit(1)

    # greedy_rr(paper_reviewer_affinities, covs, loads, args.alloc_file)
    best_revs = np.argsort(-1 * paper_reviewer_affinities, axis=0)

    complete_seln_order, partial_seln_order = get_greedy_rr(paper_reviewer_affinities, covs, loads, best_revs)
    complete_alloc, _, _ = rr(complete_seln_order, paper_reviewer_affinities, covs, loads, best_revs)
    partial_alloc, _, _ = rr(partial_seln_order, paper_reviewer_affinities, covs, loads, best_revs)

    with open("complete_order_cvpr_debug", "wb") as f:
        pickle.dump(complete_seln_order, f)
    with open("partial_order_cvpr_debug", "wb") as f:
        pickle.dump(partial_seln_order, f)
    save_alloc(complete_alloc, "complete_greedy_cvpr_debug")
    save_alloc(partial_alloc, "partial_greedy_cvpr_debug")



