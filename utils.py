import argparse
import numpy as np
import os
import pickle

from collections import Counter
from itertools import product
from sklearn import metrics


def usw(alloc, pra):
    usw = 0
    for p in alloc:
        for r in alloc[p]:
            usw += pra[r, p]
    n = pra.shape[1]
    return usw/n


def nsw(alloc, pra):
    nsw = 1
    for p in alloc:
        paper_score = 0
        for r in alloc[p]:
            paper_score += pra[r, p]
        if paper_score:
            nsw *= paper_score**(1/len(alloc))
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
                    print(paper, paper2, curr, other, alloc[paper], alloc[paper2])
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
        rev_loads.extend([ct]*num_revs)

    revs_no_load = m - len(rev_loads)
    rev_loads.extend([0]*revs_no_load)
    rev_load_dist[0] = revs_no_load

    return "%s, min: %d, max: %d, std: %.2f" % (str(rev_load_dist),
                                                min(rev_loads),
                                                max(rev_loads),
                                                np.std(rev_loads))


# Subtract the mean score of the bottom k from the mean score of the top k for k from 1 to n/2.
# Compute the AUC.
def compare_bottom_to_top(alloc, pra, covs):
    paper_scores = [get_valuation(p, alloc[p], pra)/(np.max(pra)*np.max(covs)) for p in alloc]
    paper_scores = sorted(paper_scores)

    differences = []
    end_x = int(len(alloc)/2)
    x = [i/(end_x-1) for i in range(end_x)]

    for k in range(end_x):
        differences.append(np.mean(paper_scores[-k-1:]) - np.mean(paper_scores[:k+1]))

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
    parser.add_argument("--alloc_file", type=str)
    parser.add_argument("--local_search_init_order", type=str, default=None)
    parser.add_argument("--num_processes", type=int, default=20)
    parser.add_argument("--local_search_partial_returned_order", type=str)
    return parser.parse_args()


def save_alloc(alloc, alloc_file):
    with open(alloc_file, 'wb') as f:
        pickle.dump(alloc, f)


def load_alloc(alloc_file):
    with open(alloc_file, 'rb') as f:
        return pickle.load(f)


def print_stats(alloc, paper_reviewer_affinities, covs, alg_time=0.0):
    _usw = usw(alloc, paper_reviewer_affinities)
    _nsw = nsw(alloc, paper_reviewer_affinities)
    _ef1 = ef1_violations(alloc, paper_reviewer_affinities)
    _efx = efx_violations(alloc, paper_reviewer_affinities)
    _auc = compare_bottom_to_top(alloc, paper_reviewer_affinities, covs)
    ps_min, ps_max, ps_mean, ps_std = paper_score_stats(alloc, paper_reviewer_affinities)

    # print("%0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %d & %d & %0.2f \\\\"
    #       % (alg_time, _usw, _nsw, ps_min, ps_mean, _ef1, _efx, _auc))
    print("%0.2f & %0.2f & %0.2f & %d \\\\"
          % (_usw, _nsw, ps_min, _ef1))

    # print("usw: ", _usw)
    # print("nsw: ", _nsw)
    # print("ef1 violations: ", _ef1)
    # print("efx violations: ", _efx)
    # print("auc: ", _auc)
    # print("paper coverage violations: ", paper_coverage_violations(alloc, covs))
    # print("reviewer load distribution: ", reviewer_load_distrib(alloc, paper_reviewer_affinities.shape[0]))
    # print("paper scores: ", ps_min, ps_max, ps_mean, ps_std)
    # print()


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
