import numpy as np

from collections import Counter


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


def print_stats(alloc, paper_reviewer_affinities, covs, alg_time=0.0):
    _usw = usw(alloc, paper_reviewer_affinities)
    _nsw = nsw(alloc, paper_reviewer_affinities)
    _ef1 = ef1_violations(alloc, paper_reviewer_affinities)
    _efx = efx_violations(alloc, paper_reviewer_affinities)
    ps_min, ps_max, ps_mean, ps_std = paper_score_stats(alloc, paper_reviewer_affinities)

    print("%0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %d & %d \\\\" % (alg_time, _usw, _nsw, ps_min, ps_mean, _ef1, _efx))

    print("usw: ", usw)
    print("nsw: ", nsw)
    print("ef1 violations: ", _ef1)
    print("efx violations: ", _efx)
    print("paper coverage violations: ", paper_coverage_violations(alloc, covs))
    print("reviewer load distribution: ", reviewer_load_distrib(alloc))
    print("paper scores: ", ps_min, ps_max, ps_mean, ps_std)
    print()