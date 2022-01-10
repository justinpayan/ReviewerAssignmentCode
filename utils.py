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
                    num_ef1_violations += 1

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

    return "%s, min: %d, max: %d, std: %.2f" % (str(rev_load_dist),
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="midl")
    parser.add_argument("--base_dir", type=str, default="./fair-matching/data")
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--w_value", type=float, default=0.0)
    parser.add_argument("--alloc_file", type=str, default="allocation")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--sample_size", type=int, default=-1)
    parser.add_argument("--fair_matching_dir", type=str, default="./fair-matching")
    parser.add_argument("--fairflow_timestamp", type=str, default=None)
    parser.add_argument("--fairir_timestamp", type=str, default=None)
    return parser.parse_args()


def save_alloc(alloc, alloc_file):
    with open(alloc_file, 'wb') as f:
        pickle.dump(alloc, f)


def load_alloc(alloc_file):
    with open(alloc_file, 'rb') as f:
        return pickle.load(f)


# Return the usw of running round robin on the agents in the list "seln_order"
def safe_rr_usw(seln_order, pra, covs, loads, best_revs):
    _, rev_loads_remaining, matrix_alloc = safe_rr(seln_order, pra, covs, loads, best_revs)
    _usw = np.sum(matrix_alloc * pra)
    return _usw, rev_loads_remaining, matrix_alloc


# Return the usw of running round robin on the agents in the list "seln_order".
# This is for use when we are dealing with reviewers rather than papers.
def safe_rr_revs_usw(seln_order, pra, covs, loads, best_papers):
    _, paper_covs_remaining, matrix_alloc = safe_rr_revs(seln_order, pra, covs, loads, best_papers)
    _usw = np.sum(matrix_alloc * pra)
    return _usw, paper_covs_remaining, matrix_alloc


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


"""We are trying to add r to a's bundle, but need to be sure we meet the criteria to prove the allocation is EF1. 
For all papers, we need to make sure the inductive step holds that a paper p always prefers its own assignment
to that of another paper (a, here) ~other than the 0th round with respect to p~. So we check, for any papers 
p earlier than a, do they prefer their own entire bundle to a's entire bundle? And for papers later than a, do they
prefer their own bundle to a's bundle other than the 1st reviewer? 

The first round is always safe.

Finally, we only need to check papers who have previously chosen this reviewer we are about to choose, because if 
they haven't chosen it they must have valued their own choices more. 
"""


def is_safe_choice_revs(p, r, seln_order_idx_map, matrix_alloc, revs_who_tried_papers, pra, round_num, first_paper):
    if round_num == 0 or not len(revs_who_tried_papers[p]):
        return True
    r_idx = seln_order_idx_map[r]

    # Construct the allocations we'll use for comparison
    r_alloc_orig = matrix_alloc[r, :]
    r_alloc_proposed = r_alloc_orig.copy()
    r_alloc_proposed[p] = 1
    r_alloc_proposed_reduced = r_alloc_proposed.copy()
    r_alloc_proposed_reduced[first_paper[r]] = 0

    for r_prime in revs_who_tried_papers[p]:
        if r_prime != r:
            # Check that they will not envy r if we add p to r.
            _r = r_alloc_proposed if (seln_order_idx_map[r_prime] < r_idx) else r_alloc_proposed_reduced
            v_rprime_for_r_proposed = np.sum(_r * pra[r_prime, :])

            v_rprime_for_rprime = np.sum(matrix_alloc[r_prime, :] * pra[r_prime, :])
            if v_rprime_for_r_proposed > v_rprime_for_rprime and not np.isclose(v_rprime_for_r_proposed, v_rprime_for_rprime):
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
                return alloc, loads_copy, matrix_alloc
    return alloc, loads_copy, matrix_alloc


# To be used for assigning papers to reviewers, rather than the other way around.
def safe_rr_revs(seln_order, pra, covs, loads, best_papers):
    alloc = {r: list() for r in seln_order}
    matrix_alloc = np.zeros((pra.shape), dtype=np.bool)

    # loads_copy = loads.copy()
    covs_copy = covs.copy()

    # When selecting a paper, you need to check for EF1 (inductive step) with all other reviewers who either
    # previously chose that paper or were themselves forced to pass over that paper... aka anyone
    # that ever TRIED to pick that paper. A reviewer that never
    # tried to pick that paper will have picked one they liked better anyway.
    revs_who_tried_papers = defaultdict(list)
    first_paper = {}

    seln_order_idx_map = {r: idx for idx, r in enumerate(seln_order)}
    num_rounds = np.sum(covs)/loads.shape[0]

    for round_num in range(num_rounds):
        for r in seln_order:
            # If there are a lot of reviewers, we may assign all papers at some point in the final round.
            if np.sum(covs_copy):
                new_assn = False
                for p in best_papers[r, :]:
                    if covs_copy[p] > 0 and p not in alloc[r]:
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
                    return alloc, loads_copy, matrix_alloc
    return alloc, loads_copy, matrix_alloc


def print_stats(alloc, paper_reviewer_affinities, covs, alg_time=0.0):
    _usw = usw(alloc, paper_reviewer_affinities)
    _nsw = nsw(alloc, paper_reviewer_affinities)
    envy = total_envy(alloc, paper_reviewer_affinities)
    _ef1 = ef1_violations(alloc, paper_reviewer_affinities)
    _efx = efx_violations(alloc, paper_reviewer_affinities)
    _gini = compute_gini(alloc, paper_reviewer_affinities)
    _mean_bottom_ten, _std_bottom_ten = get_percentile_mean_std(alloc, paper_reviewer_affinities, .10)
    _mean_bottom_quartile, _std_bottom_quartile = get_percentile_mean_std(alloc, paper_reviewer_affinities, .25)
    ps_min, ps_max, ps_mean, ps_std = paper_score_stats(alloc, paper_reviewer_affinities)

    print("%0.2f & %0.2f & %0.2f & %d \\\\"
          % (_usw, _nsw, ps_min, _ef1))

    print("envy: ", envy)
    print("gini: ", _gini)
    print("mean, std 10-percentile: ", _mean_bottom_ten, _std_bottom_ten)
    print("mean, std 25-percentile: ", _mean_bottom_quartile, _std_bottom_quartile)


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    alloc_file = args.alloc_file
    base_dir = args.base_dir

    alloc = load_alloc(alloc_file)

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)

    print_stats(alloc, paper_reviewer_affinities, covs)
