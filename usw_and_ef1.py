# Specify a problem by passing in the valuation functions
# Also pass in groups of items that are identical

# The algorithm will assume that a good's marginal contribution is 0 if an identical item is already in the bag

# As stated in Benabbou 20 (?), the algorithm first finds a max USW allocation by optimizing over an intersection
# of many matroids (presumably I can use the ellipsoid method, either by implementing or by leveraging an API).

# We then make transfers of items until no one envies anyone else more than 1 item.

# This only works for (0,1)-SUB valuation functions.

import numpy as np
import sys
import time

from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow


def construct_graph(reviewer_loads, paper_reviewer_affinities, paper_capacities):
    # 1d numpy array, 2d array, 1d numpy array
    reviewers = list(range(reviewer_loads.shape[0]))
    reviewers = [i + 2 for i in reviewers]
    papers = list(range(paper_capacities.shape[0]))
    papers = [i + len(reviewers) + 2 for i in papers]

    n = len(reviewers) + len(papers) + 2
    graph = csr_matrix(np.zeros((n, n), dtype=np.int32))
    # Draw edges from reviewers to papers
    graph[2:2+len(reviewers), 2+len(reviewers):] = paper_reviewer_affinities
    # Draw edges from source to reviewers
    graph[0, 2:2+len(reviewers)] = reviewer_loads
    # Draw edges from papers to sink
    graph[2+len(reviewers):, 1] = paper_capacities.reshape(-1, 1)

    return graph


def get_alloc_from_flow_result(flow, reviewer_loads, paper_capacities):
    # Loop over the reviewers, and for every paper where there is a 1, add it to that paper's set.
    reviewers = list(range(reviewer_loads.shape[0]))
    reviewers = [i + 2 for i in reviewers]
    papers = list(range(paper_capacities.shape[0]))
    papers = [i + len(reviewers) + 2 for i in papers]

    alloc = {p-len(reviewers)-2: set() for p in papers}

    for r in reviewers:
        for p in papers:
            if flow[r, p] == 1:
                alloc[p-len(reviewers)-2].add(r-2)

    return alloc


# Make sure the reviewer loads and paper capacities are in the same order as the papers and reviewers in the matrix.
# paper_reviewer_affinities -> Rows correspond to reviewers and columns correspond to papers
def get_usw_alloc(reviewer_loads, paper_reviewer_affinities, paper_capacities):
    # 1d numpy array, 2d array, 1d numpy array
    g = construct_graph(reviewer_loads, paper_reviewer_affinities, paper_capacities)

    f = maximum_flow(g, 0, 1)
    # flow_value = f.flow_value
    residuals = f.residual

    alloc = get_alloc_from_flow_result(residuals, reviewer_loads, paper_capacities)

    return alloc


def get_valuation(paper, reviewer_set, paper_reviewer_affinities):
    val = 0
    for r in reviewer_set:
        val += paper_reviewer_affinities[r, paper]
    return val


def make_transfer(p_from, p_to, alloc, paper_reviewer_affinities):
    revs_from = alloc[p_from]
    revs_to = alloc[p_to]

    for r in revs_from:
        if paper_reviewer_affinities[r, p_to] == 1 and r not in revs_to:
            alloc[p_to].add(r)
            alloc[p_from].remove(r)
            break

    return alloc


def eits(alloc, paper_reviewer_affinities):
    # While there is an envious pair, make the swap
    had_envy = True
    while had_envy:
        had_envy = False
        for paper in alloc:
            for paper2 in alloc:
                if paper != paper2:
                    other = get_valuation(paper, alloc[paper2], paper_reviewer_affinities) - 1
                    curr = get_valuation(paper, alloc[paper], paper_reviewer_affinities)
                    if other > curr:
                        had_envy = True
                        alloc = make_transfer(paper2, paper, alloc, paper_reviewer_affinities)
    return alloc


def binarize(paper_reviewer_affinities, threshold):
    new_pra = np.copy(paper_reviewer_affinities)
    new_pra[paper_reviewer_affinities >= threshold] = 1
    new_pra[paper_reviewer_affinities < threshold] = 0

    # This part is kind of cheating, but we should make sure there are enough suitable reviewers per paper.
    # Obviously avoiding this kind of hack is a good reason to try to come up with an algorithm that'll work for
    # the original affinities.
    for p in range(new_pra.shape[1]):
        if np.sum(new_pra[:, p]) < 3:
            new_pra[np.argsort(new_pra[:, p])[::-1][:3], p] = 1

    return new_pra


def max_usw_and_ef1_alloc(reviewer_loads, paper_reviewer_affinities_bin, paper_capacities, threshold):
    max_usw = get_usw_alloc(reviewer_loads, paper_reviewer_affinities_bin, paper_capacities)
    # print(max_usw)
    ef1_usw = eits(max_usw, paper_reviewer_affinities_bin)
    # print(ef1_usw)
    # for p in max_usw:
    #     if ef1_usw[p] != max_usw[p]:
    #         print(p, max_usw[p], ef1_usw[p])
    return ef1_usw


# ef1_violations_set = set()


def efx_violations(alloc, pra):
    num_efx_violations = 0
    for paper in alloc:
        for paper2 in alloc:
            if paper != paper2:
                other = get_valuation(paper, alloc[paper2], pra)
                curr = get_valuation(paper, alloc[paper], pra)
                for reviewer in alloc[paper2]:
                    if other - pra[reviewer, paper] > curr:
                        num_efx_violations += 1
                        break

    return num_efx_violations


def ef1_violations(alloc, pra):
    # print("ef1 violations")
    num_ef1_violations = 0
    for paper in alloc:
        for paper2 in alloc:
            if paper != paper2:
                other = get_valuation(paper, alloc[paper2], pra)
                curr = get_valuation(paper, alloc[paper], pra)
                found_reviewer_to_drop = False
                if alloc[paper2]:
                    for reviewer in alloc[paper2]:
                        if other - pra[reviewer, paper] <= curr:
                            found_reviewer_to_drop = True
                            break
                    if not found_reviewer_to_drop:
                        num_ef1_violations += 1

    return num_ef1_violations


def usw(alloc, pra):
    usw = 0
    for p in alloc:
        for r in alloc[p]:
            usw += pra[r, p]
    return usw


def nsw(alloc, pra):
    nsw = 1
    for p in alloc:
        paper_score = 0
        for r in alloc[p]:
            paper_score += pra[r, p]
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
    return "Min: %.3f, Max: %.3f, Mean: %.3f, Std: %.3f" % (min(paper_scores),
                                                            max(paper_scores),
                                                            np.mean(paper_scores),
                                                            np.std(paper_scores))


def reviewer_load_distrib(alloc):
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

    return "%s, min: %d, max: %d, std: %.2f" % (str(rev_load_dist),
                                                min(rev_loads),
                                                max(rev_loads),
                                                np.std(rev_loads))


def paper_coverage_violations(alloc, covs):
    viol = 0
    for paper, revs in alloc.items():
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


def print_stats(alloc, bin_pra, paper_reviewer_affinities, covs):
    print("bin usw: ", usw(alloc, bin_pra))
    print("bin nsw: ", nsw(alloc, bin_pra))
    print("bin ef1 violations: ", ef1_violations(alloc, bin_pra))
    print("bin efx violations: ", efx_violations(alloc, bin_pra))
    print("usw: ", usw(alloc, paper_reviewer_affinities))
    print("nsw: ", nsw(alloc, paper_reviewer_affinities))
    print("ef1 violations: ", ef1_violations(alloc, paper_reviewer_affinities))
    print("efx violations: ", efx_violations(alloc, paper_reviewer_affinities))
    print("paper coverage violations: ", paper_coverage_violations(alloc, covs))
    print("reviewer load distribution: ", reviewer_load_distrib(alloc))
    print("paper scores: ", paper_score_stats(alloc, paper_reviewer_affinities))
    print()


if __name__ == "__main__":
    # Load the data for a conference, run the algorithm. Do something with the result...
    # Eventually we will run some analysis on it. Right now, maybe just print out the reviewers assigned to the
    # papers, the usw, etc.

    # Let's first start with MIDL with a .3 threshold?
    dataset = sys.argv[1]
    timestamps = {"cvpr": "2020-09-16-10-28-42", "cvpr2018": "2020-09-16-10-26-09", "midl": "2020-09-16-09-32-53"}

    paper_reviewer_affinities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/scores.npy" % dataset)
    reviewer_loads = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/loads.npy" % dataset)
    paper_capacities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/covs.npy" % dataset)
    threshold = 0.4

    bin_pra = binarize(paper_reviewer_affinities, threshold)
    start = time.time()
    alloc = max_usw_and_ef1_alloc(reviewer_loads, bin_pra, paper_capacities, threshold)
    runtime = time.time() - start

    print("EIT-based Algorithm Results (%.2f)" % threshold)
    print("%.2f seconds" % runtime)
    print_stats(alloc, bin_pra, paper_reviewer_affinities, paper_capacities)

    # Load the fairflow solution for MIDL without the reviewer lower bounds...
    fairflow_soln = load_fairflow_soln("/home/justinspayan/Fall_2020/fair-matching/exp_out/%s/fairflow/"
                                       "%s/results/assignment.npy" % (dataset, timestamps[dataset]))
    print("Fairflow Results")
    print_stats(fairflow_soln, bin_pra, paper_reviewer_affinities, paper_capacities)
    print("***********\n***********\n")


