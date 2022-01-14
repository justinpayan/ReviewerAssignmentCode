import argparse
import math
import numpy as np
import os
import pickle
import random

from gini import *
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product, permutations
from sklearn import metrics

from tqdm import tqdm


def usw(alloc, pra):
    usw = 0
    for p in alloc:
        for r in alloc[p]:
            usw += pra[r, p]
    n = pra.shape[1]
    return usw / n


def nsw(alloc, pra):
    nsw = 1
    for p in alloc:
        paper_score = 0
        for r in alloc[p]:
            paper_score += pra[r, p]
        if paper_score:
            nsw *= paper_score ** (1 / len(alloc))
    return nsw


def paper_score_stats(alloc, paper_reviewer_affinities):
    # Min, max, mean, std
    paper_scores = []
    for p, assigned_revs in alloc.items():
        paper_score = 0.0
        for r in assigned_revs:
            paper_score += paper_reviewer_affinities[r, p]
        paper_scores.append(paper_score)
    return min(paper_scores), max(paper_scores), np.mean(paper_scores), np.std(paper_scores)
    # return "Min: %.3f, Max: %.3f, Mean: %.3f, Std: %.3f" % (min(paper_scores),
    #                                                         max(paper_scores),
    #                                                         np.mean(paper_scores),
    #                                                         np.std(paper_scores))


def get_valuation(paper, reviewer_set, paper_reviewer_affinities):
    val = 0
    for r in reviewer_set:
        val += paper_reviewer_affinities[r, paper]
    return val


def efx_violations(alloc, pra):
    num_efx_violations = 0
    n = pra.shape[1]
    for paper, paper2 in product(range(n), range(n)):
        if paper != paper2:
            other = get_valuation(paper, alloc[paper2], pra)
            curr = get_valuation(paper, alloc[paper], pra)
            for reviewer in alloc[paper2]:
                val_dropped = other - pra[reviewer, paper]
                if val_dropped > curr and not np.isclose(val_dropped, curr):
                    num_efx_violations += 1
                    break

    return num_efx_violations


def total_envy(alloc, scores):
    envy = 0.0
    n = scores.shape[1]
    for paper, paper2 in product(range(n), range(n)):
        if paper != paper2:
            other = get_valuation(paper, alloc[paper2], scores)
            curr = get_valuation(paper, alloc[paper], scores)
            envy += max([other - curr, 0])
    return envy


def ef1_violations(alloc, pra):
    num_ef1_violations = 0
    n = pra.shape[1]
    for paper, paper2 in product(range(n), range(n)):
        if paper != paper2:
            other = get_valuation(paper, alloc[paper2], pra)
            curr = get_valuation(paper, alloc[paper], pra)
            found_reviewer_to_drop = False
            if alloc[paper2]:
                for reviewer in alloc[paper2]:
                    val_dropped = other - pra[reviewer, paper]
                    if val_dropped < curr or np.abs(val_dropped - curr) < 1e-8:
                        found_reviewer_to_drop = True
                        break
                if not found_reviewer_to_drop:
                    # print(paper, paper2, curr, other, alloc[paper], alloc[paper2])
                    num_ef1_violations += 1

    # alternative. Requires a ton of memory.
    # matrix_alloc = np.zeros((pra.shape))
    # m, n = pra.shape
    # for a in alloc:
    #     for r in alloc[a]:
    #         matrix_alloc[r, a] = 1
    # W = np.dot(pra.transpose(), matrix_alloc)
    # # I want n matrices, where each matrix i has rows j which hold the value of i for items given to j.
    # # M = (self.scores.reshape((1, self.m, self.n)) * allocation.reshape((self.m, 1, self.n))).transpose(0, 1)
    # # M = torch.max(M, dim=2).values
    # M = pra.transpose().reshape(n, m, 1) * matrix_alloc.reshape(1, m, n)
    #
    # M = np.max(M, axis=1)
    # # M's ij element is most valuable item for i in j's bundle. If we just change it so that all items not owned
    # # by j are worth 1000 and then take the min, this will be EFX instead of EF1.
    # strong_envy = (W - M) - np.diag(W).reshape((-1, 1))
    # strong_envy = np.maximum(strong_envy, 0)
    # diag_mask = np.ones((n, n)) - np.eye(n)
    # strong_envy = strong_envy * diag_mask
    # num_ef1_violations = np.sum(strong_envy > 1e-6)
    #
    # return num_ef1_violations

    return num_ef1_violations


def paper_coverage_violations(alloc, covs):
    viol = 0
    for paper in range(covs.shape[0]):
        revs = alloc[paper]
        if len(revs) < covs[paper]:
            viol += 1
    return viol


def load_fairflow_soln(path_to_output):
    ffs = np.load(path_to_output)

    reviewers = list(range(ffs.shape[0]))
    papers = list(range(ffs.shape[1]))

    alloc = {p: set() for p in papers}

    for r in reviewers:
        for p in papers:
            if ffs[r, p] == 1:
                alloc[p].add(r)

    return alloc


def reviewer_load_distrib(alloc, m):
    rev_loads = Counter()
    for rev_set in alloc.values():
        for r in rev_set:
            rev_loads[r] += 1
    rev_load_dist = Counter()
    for ct in rev_loads.values():
        rev_load_dist[ct] += 1

    rev_loads = []
    for ct, num_revs in rev_load_dist.items():
        rev_loads.extend([ct] * num_revs)

    revs_no_load = m - len(rev_loads)
    rev_loads.extend([0] * revs_no_load)
    rev_load_dist[0] = revs_no_load

    return "%s, min: %d, max: %d, std: %.2f" % (str(sorted(rev_load_dist.items())),
                                                min(rev_loads),
                                                max(rev_loads),
                                                np.std(rev_loads))


def get_percentile_mean_std(alloc, scores, perc):
    paper_scores = [get_valuation(p, alloc[p], scores) for p in alloc]
    paper_scores = sorted(paper_scores)
    print(paper_scores[:20])
    percentile_scores = paper_scores[:math.floor(perc*len(paper_scores))]
    return np.mean(percentile_scores), np.std(percentile_scores)


def compute_gini(alloc, scores):
    paper_scores = [get_valuation(p, alloc[p], scores) for p in alloc]
    paper_scores = sorted(paper_scores)
    return gini(np.array(paper_scores))


# Subtract the mean score of the bottom k from the mean score of the top k for k from 1 to n/2.
# Compute the AUC.
def compare_bottom_to_top(alloc, pra, covs):
    paper_scores = [get_valuation(p, alloc[p], pra) / (np.max(pra) * np.max(covs)) for p in alloc]
    paper_scores = sorted(paper_scores)

    differences = []
    end_x = int(len(alloc) / 2)
    x = [i / (end_x - 1) for i in range(end_x)]

    for k in range(end_x):
        differences.append(np.mean(paper_scores[-k - 1:]) - np.mean(paper_scores[:k + 1]))

    # print(x)
    # print(differences)

    return metrics.auc(x, differences)


# Compute the max number of swaps you can make between envious pairs without causing anyone to drop below the
# mean. Obviously this is flawed because if you just have a lower mean it is easier to make swaps. But actually that
# might kind of balance it out. And it begs the question of like why not just minimize this directly?
def number_of_envy_swaps(alloc, pra):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="midl")
    parser.add_argument("--base_dir", type=str, default="/home/justinspayan/Fall_2020/fair-matching/data")
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--w_value", type=float, default=0.0)
    parser.add_argument("--alloc_file", type=str)
    parser.add_argument("--local_search_init_order", type=str, default=None)
    parser.add_argument("--num_processes", type=int, default=20)
    parser.add_argument("--local_search_partial_returned_order", type=str)
    parser.add_argument("--mg_file", type=str, default=None)
    parser.add_argument("--num_distrib_jobs", type=int, default=1000)
    parser.add_argument("--job_num", type=int, default=0)
    parser.add_argument('--init_run', dest='init_run', action='store_true')
    parser.add_argument('--no_init_run', dest='init_run', action='store_false')
    parser.set_defaults(init_run=False)
    return parser.parse_args()


def save_alloc(alloc, alloc_file):
    with open(alloc_file, 'wb') as f:
        pickle.dump(alloc, f)


def load_alloc(alloc_file):
    with open(alloc_file, 'rb') as f:
        return pickle.load(f)


""" Check all ways to complete the tuple set, create the total ordering for each one and compute usw
    from running the safe_rr_usw function. Return the best usw. DO NOT try to run on small sets."""


def max_safe_rr_usw(tuple_set, pra, covs, loads, best_revs, normalizer=1.0):
    m, n = pra.shape

    agents = [x[0] for x in tuple_set]
    positions = [x[1] for x in tuple_set]

    remaining_positions = list(set(range(n)) - set(positions))

    usws = []

    remaining_agents = set(range(n)) - set(agents)

    for remaining_agent_order in tqdm(permutations(remaining_agents), total=math.factorial(len(remaining_agents))):
        S = tuple_set | set(zip(remaining_agent_order, remaining_positions))

        seln_order = [x[0] for x in sorted(S, key=lambda x: x[1])]
        usws.append(safe_rr_usw(seln_order, pra, covs, loads, best_revs)[0])

    # x = (np.max(usws) / normalizer) * np.log(len(tuple_set))
    x = (np.max(usws)/normalizer) * (len(tuple_set))
    # return x * (1-(1-(len(tuple_set)/n))**n)
    # return x * len(tuple_set)**30
    return x


""" Sample a bunch of ways to complete the tuple set, create the total ordering for each one and compute usw
    from running the safe_rr_usw function."""


def estimate_expected_safe_rr_usw(tuple_set, pra, covs, loads, best_revs, n_iters=100, normalizer=1.0):
    m, n = pra.shape

    agents = [x[0] for x in tuple_set]
    positions = [x[1] for x in tuple_set]

    remaining_positions = set(range(n)) - set(positions)

    usws = []
    for _ in range(n_iters):
        S = deepcopy(tuple_set)
        remaining_agents = sorted(list(set(range(n)) - set(agents)), key=lambda x: random.random())

        for p in remaining_positions:
            S.add((remaining_agents.pop(), p))

        seln_order = [x[0] for x in sorted(S, key=lambda x: x[1])]
        usws.append(safe_rr_usw(seln_order, pra, covs, loads, best_revs)[0])

    x = (np.mean(usws)/normalizer) * len(tuple_set)**2
    # return x * (1-(1-(len(tuple_set)/n))**n)
    # return x * len(tuple_set)**30
    return x


# Return the usw of running round robin on the agents in the list "seln_order"
def safe_rr_usw(seln_order, pra, covs, loads, best_revs):
    # rr_alloc, rev_loads_remaining, matrix_alloc = rr(seln_order, pra, covs, loads, best_revs)
    _, rev_loads_remaining, matrix_alloc = safe_rr(seln_order, pra, covs, loads, best_revs)
    # _usw = usw(rr_alloc, pra)
    _usw = np.sum(matrix_alloc * pra)
    # print("USW ", time.time() - start)
    return _usw, rev_loads_remaining, matrix_alloc


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
        # best_revs = np.argmax(pra[:, seln_order] * (matrix_alloc[:, seln_order] == 0) * \
        # (loads_copy > 0).reshape((-1,1)), axis=0)

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
        # TODO: This code also doesn't consider that we are operating over a subset of the papers
        # TODO: I think we just need to fix that by setting all non-considered papers' reviewers in which_revs
        # TODO: to -1. Of course, it also isn't considering the selection order at all. But if we can
        # TODO: modify this code so that it knows which agents we're concerned about and in what order,
        # TODO: then 1) it might actually speed up RR and 2) it might be possible to feed in a bunch of
        # TODO: those order arrays and run RR for hundreds or thousands of orders in parallel.
        # TODO: Maybe we can actually run the whole thing on the GPU and use torch in that case?
        # TODO: So there won't be any loss computation, just good old-fashioned cuda BLAS operations.
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
        #         # we need to set the first which_agents_to_change nonzero elements of each row to 0
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
        #         #     violating_agents = np.where(which_revs == \
        #         violated_reviewers[idx])[0][which_agents_to_change[idx]:]
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


"""We are trying to add r to a's bundle, but need to be sure we meet the criteria to prove the allocation is EF1. 
For all papers, we need to make sure the inductive step holds that a paper p always prefers its own assignment
to that of another paper (a, here) ~other than the 0th round with respect to p~. So we check, for any papers 
p earlier than a, do they prefer their own entire bundle to a's entire bundle? And for papers later than a, do they
prefer their own bundle to a's bundle other than the 1st reviewer? 

The first round is always safe.

Finally, we only need to check papers who have previously chosen this reviewer we are about to choose, because if 
they haven't chosen it they must have valued their own choices more. 
"""


def is_safe_choice(r, a, seln_order_idx_map, matrix_alloc, papers_who_tried_revs, pra, round_num, first_reviewer):
    if round_num == 0 or not len(papers_who_tried_revs[r]):
        return True
    a_idx = seln_order_idx_map[a]

    # Construct the allocations we'll use for comparison
    a_alloc_orig = matrix_alloc[:, a]
    a_alloc_proposed = a_alloc_orig.copy()
    a_alloc_proposed[r] = 1
    a_alloc_proposed_reduced = a_alloc_proposed.copy()
    a_alloc_proposed_reduced[first_reviewer[a]] = 0

    for p in papers_who_tried_revs[r]:
        if p != a:
            # Check that they will not envy a if we add r to a.
            _a = a_alloc_proposed if (seln_order_idx_map[p] < a_idx) else a_alloc_proposed_reduced
            v_p_for_a_proposed = np.sum(_a * pra[:, p])

            v_p_for_p = np.sum(matrix_alloc[:, p] * pra[:, p])
            if v_p_for_a_proposed > v_p_for_p and not np.isclose(v_p_for_a_proposed, v_p_for_p):
                return False
    return True


def safe_rr(seln_order, pra, covs, loads, best_revs):
    alloc = {p: list() for p in seln_order}
    matrix_alloc = np.zeros((pra.shape), dtype=np.bool)

    loads_copy = loads.copy()

    # When selecting a reviewer, you need to check for EF1 (inductive step) with all other papers who either
    # previously chose that reviewer or were themselves forced to pass over that reviewer... aka anyone
    # that ever TRIED to pick that reviewer. A paper that never
    # tried to pick that reviewer will have picked someone they liked better anyway.
    papers_who_tried_revs = defaultdict(list)
    first_reviewer = {}

    seln_order_idx_map = {p: idx for idx, p in enumerate(seln_order)}

    # Assume all covs are the same
    for round_num in range(covs[seln_order[0]]):
        for a in seln_order:
            new_assn = False
            for r in best_revs[:, a]:
                if loads_copy[r] > 0 and r not in alloc[a]:
                    if is_safe_choice(r, a, seln_order_idx_map, matrix_alloc,
                                      papers_who_tried_revs, pra, round_num, first_reviewer):
                        loads_copy[r] -= 1
                        alloc[a].append(r)
                        matrix_alloc[r, a] = 1
                        if round_num == 0:
                            first_reviewer[a] = r
                        papers_who_tried_revs[r].append(a)
                        new_assn = True
                        break
                    else:
                        papers_who_tried_revs[r].append(a)
            if not new_assn:
                print("no new assn")
                sys.exit(0)
    return alloc, loads_copy, matrix_alloc


def print_stats(alloc, paper_reviewer_affinities, covs, alg_time=0.0):
    # _usw = usw(alloc, paper_reviewer_affinities)
    # _nsw = nsw(alloc, paper_reviewer_affinities)
    # envy = total_envy(alloc, paper_reviewer_affinities)
    # _ef1 = ef1_violations(alloc, paper_reviewer_affinities)
    # _efx = efx_violations(alloc, paper_reviewer_affinities)
    # _auc = compare_bottom_to_top(alloc, paper_reviewer_affinities, covs)
    # _gini = compute_gini(alloc, paper_reviewer_affinities)
    # _mean_bottom_ten, _std_bottom_ten = get_percentile_mean_std(alloc, paper_reviewer_affinities, .10)
    # _mean_bottom_quartile, _std_bottom_quartile = get_percentile_mean_std(alloc, paper_reviewer_affinities, .25)
    # ps_min, ps_max, ps_mean, ps_std = paper_score_stats(alloc, paper_reviewer_affinities)

    # # print("%0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %d & %d & %0.2f \\\\"
    # #       % (alg_time, _usw, _nsw, ps_min, ps_mean, _ef1, _efx, _auc))
    # print("%0.2f & %0.2f & %0.2f & %d \\\\"
    #       % (_usw, _nsw, ps_min, _ef1))

    # # print("auc: ", _auc)
    # print("envy: ", envy)
    # print("gini: ", _gini)
    # print("mean, std 10-percentile: ", _mean_bottom_ten, _std_bottom_ten)
    # print("mean, std 25-percentile: ", _mean_bottom_quartile, _std_bottom_quartile)

    # Number of papers per reviewer
    m, _ = paper_reviewer_affinities.shape
    print(m)
    print(np.max(paper_reviewer_affinities))
    print(reviewer_load_distrib(alloc, m))

    paper_reviewer_affinities = paper_reviewer_affinities - np.max(paper_reviewer_affinities)
    # Compute USW, NSW, worst burden, and num EF1 violations for REVIEWERS

    usw = 0
    rev_scores = defaultdict(int)
    reverse_alloc = defaultdict(list)
    for p in alloc:
        for r in alloc[p]:
            usw += paper_reviewer_affinities[r, p]
            rev_scores[r] += paper_reviewer_affinities[r, p]
            reverse_alloc[r].append(p)
    print("USW (revs): ", usw / m)

    all_rev_scores = list(rev_scores.values())

    nsw = 1
    for r in rev_scores:
        if rev_scores[r]:
            nsw *= ((-1*rev_scores[r])**(1/m))
    print("NSW (revs) ", -1*nsw)

    print("Min score (revs): ", np.min(all_rev_scores))

    for perc in [.1, .25]:
        _rev_scores = [rev_scores[r] for r in range(m)]
        _rev_scores = sorted(_rev_scores)
        percentile_scores = _rev_scores[:math.floor(perc * m)]
        print("Mean std at ", perc, np.mean(percentile_scores), np.std(percentile_scores))

    num_ef1_violations = 0
    for rev, rev2 in product(range(m), range(m)):
        if rev != rev2:
            if reverse_alloc[rev]:
                other = np.sum([paper_reviewer_affinities[rev, p] for p in reverse_alloc[rev2]])
                curr = np.sum([paper_reviewer_affinities[rev, p] for p in reverse_alloc[rev]])
                found_paper_to_drop = False
                for p in reverse_alloc[rev]:
                    val_dropped = curr - paper_reviewer_affinities[rev, p]
                    # print(curr)
                    # print(val_dropped)
                    # print(other)
                    # print(val_dropped >= other)
                    if val_dropped >= other or np.abs(val_dropped - other) < 1e-8:
                        found_paper_to_drop = True
                        break
                if not found_paper_to_drop:
                    # print(rev, rev2, reverse_alloc[rev], reverse_alloc[rev2], curr, other)
                    # print(paper, paper2, curr, other, alloc[paper], alloc[paper2])
                    num_ef1_violations += 1
    print("Number of EF1 violations: ", num_ef1_violations)


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    alloc_file = args.alloc_file
    base_dir = args.base_dir

    alloc = load_alloc(alloc_file)

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    # print(sorted(alloc))
    print_stats(alloc, paper_reviewer_affinities, covs)
