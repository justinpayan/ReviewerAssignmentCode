import argparse
import math
import numpy as np

from collections import Counter, defaultdict
from itertools import product


def usw(alloc, pra):
    usw = 0
    for p in alloc:
        for r in alloc[p]:
            usw += pra[r, p]
    n = pra.shape[1]
    return usw / n


def nsw(alloc, pra):
    _nsw = 1
    for p in alloc:
        paper_score = 0
        for r in alloc[p]:
            paper_score += pra[r, p]
        if paper_score:
            _nsw *= paper_score ** (1 / len(alloc))
    return _nsw


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
            # other = get_valuation(paper, alloc[paper2], pra) - np.max(pra[alloc[paper2], [paper2] * len(alloc[paper2])])
            # curr = get_valuation(p, alloc[p], scores)

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


def reviewer_ef1_violations(paper_reviewer_affinities, m, reverse_alloc):
    num_ef1_violations = 0
    for rev, rev2 in product(range(m), range(m)):
        if rev != rev2:
            if reverse_alloc[rev]:
                other = np.sum([paper_reviewer_affinities[rev, p] for p in reverse_alloc[rev2]])
                curr = np.sum([paper_reviewer_affinities[rev, p] for p in reverse_alloc[rev]])
                found_paper_to_drop = False
                for p in reverse_alloc[rev]:
                    val_dropped = curr - paper_reviewer_affinities[rev, p]

                    if val_dropped >= other or np.abs(val_dropped - other) < 1e-8:
                        found_paper_to_drop = True
                        break
                if not found_paper_to_drop:
                    num_ef1_violations += 1

    return num_ef1_violations


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

    return "%s, min: %d, max: %d, std: %.2f" % \
           (str(rev_load_dist), min(rev_loads), max(rev_loads), np.std(rev_loads))


def get_percentile_mean_std(alloc, scores, perc):
    paper_scores = [get_valuation(p, alloc[p], scores) for p in alloc]
    paper_scores = sorted(paper_scores)
    print(paper_scores[:20])
    percentile_scores = paper_scores[:math.floor(perc*len(paper_scores))]
    return np.mean(percentile_scores), np.std(percentile_scores)


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def compute_gini(alloc, scores):
    paper_scores = [get_valuation(p, alloc[p], scores) for p in alloc]
    paper_scores = sorted(paper_scores)
    return gini(np.array(paper_scores))


def print_stats(alloc, paper_reviewer_affinities):
    _usw = usw(alloc, paper_reviewer_affinities)
    _nsw = nsw(alloc, paper_reviewer_affinities)
    envy = total_envy(alloc, paper_reviewer_affinities)
    _ef1 = ef1_violations(alloc, paper_reviewer_affinities)
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

    print("Reviewer load distrib: ", reviewer_load_distrib(alloc, paper_reviewer_affinities.shape[0]))

    m, _ = paper_reviewer_affinities.shape

    paper_reviewer_affinities = paper_reviewer_affinities - np.max(paper_reviewer_affinities)
    # Compute USW, NSW, worst burden, and num EF1 violations for REVIEWERS

    rusw = 0
    rev_scores = defaultdict(int)
    reverse_alloc = defaultdict(list)
    for p in alloc:
        for r in alloc[p]:
            rusw += paper_reviewer_affinities[r, p]
            rev_scores[r] += paper_reviewer_affinities[r, p]
            reverse_alloc[r].append(p)
    print("USW (revs): ", rusw / m)

    all_rev_scores = list(rev_scores.values())

    rnsw = 1
    for r in rev_scores:
        if rev_scores[r]:
            rnsw *= ((-1 * rev_scores[r]) ** (1 / m))
    print("NSW (revs) ", -1 * rnsw)

    print("Min score (revs): ", np.min(all_rev_scores))

    for perc in [.1, .25]:
        _rev_scores = [rev_scores[r] for r in range(m)]
        _rev_scores = sorted(_rev_scores)
        percentile_scores = _rev_scores[:math.floor(perc * m)]
        print("Mean std (revs) at ", perc, np.mean(percentile_scores), np.std(percentile_scores))

    rev_ef1_viols = reviewer_ef1_violations(paper_reviewer_affinities, m, reverse_alloc)
    print("Number of reviewer EF1 violations: ", rev_ef1_viols)


def load_soln_from_npy(path_to_output):
    ffs = np.load(path_to_output)

    reviewers = list(range(ffs.shape[0]))
    papers = list(range(ffs.shape[1]))

    alloc = {p: set() for p in papers}

    for r in reviewers:
        for p in papers:
            if ffs[r, p] == 1:
                alloc[p].add(r)

    return alloc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alloc_file", type=str)
    parser.add_argument("--pra_file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    alloc_file = args.alloc_file
    pra_file = args.pra_file

    alloc = load_soln_from_npy(alloc_file)
    paper_reviewer_affinities = np.load(pra_file)

    print_stats(alloc, paper_reviewer_affinities)
