import os
import random
import time

from utils import *


def schedule_by_random(scores, loads, covs, ordering=None):
    m, n = scores.shape
    best_revs = np.argsort(-1 * scores, axis=0)
    if ordering is None:
        ordering = sorted(range(n), key=lambda x: random.random())

    a, _, _ = safe_rr(ordering, scores, covs, loads, best_revs)
    # a, _, _ = rr(ordering, scores, covs, loads, best_revs)
    return a


def run_algo(dataset, base_dir):
    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)

    alloc = schedule_by_random(paper_reviewer_affinities, reviewer_loads, paper_capacities)
    return alloc


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = args.base_dir
    alloc_file = args.alloc_file

    np.random.seed(args.seed)
    random.seed(args.seed)

    times = []
    usws = []
    nsws = []
    ef1s = []
    efxs = []
    ps_mins = []
    ps_maxs = []
    ps_means = []
    ps_stds = []
    # all_stats = [times, usws, nsws, ef1s, efxs, ps_mins, ps_maxs, ps_means, ps_stds]
    all_stats = [usws, nsws, ps_mins, ef1s]

    for _ in range(10):
        start = time.time()
        alloc = run_algo(dataset, base_dir)
        runtime = time.time() - start
        times.append(runtime)

        # alloc = load_alloc(alloc_file)
        # save_alloc(alloc, alloc_file)

        # print("Barman Algorithm Results")
        # print("%.2f seconds" % runtime)
        # print_stats(alloc, paper_reviewer_affinities, paper_capacities)

        print("Random RR Results")
        print("%.2f seconds" % runtime)

        scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
        loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
        covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))

        # print(alloc)
        # print_stats(alloc, paper_reviewer_affinities, paper_capacities)
        usws.append(usw(alloc, scores))
        nsws.append(nsw(alloc, scores))
        ef1s.append(ef1_violations(alloc, scores))
        print(ef1s[-1])
        efxs.append(efx_violations(alloc, scores))
        pmi, pma, pme, pst = paper_score_stats(alloc, scores)
        ps_mins.append(pmi)
        ps_maxs.append(pma)
        ps_means.append(pme)
        ps_stds.append(pst)
    # ENDFOR

    means = [np.mean(stat) for stat in all_stats]
    stds = [np.std(stat) for stat in all_stats]
    print(" & ".join(["$%0.2f \\pm %0.2f$" % (m, s) for (m, s) in zip(means, stds)]))

    # for stat in all_stats:
    #     print(np.mean(stat))


