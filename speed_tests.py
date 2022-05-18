import argparse
import numpy as np
import os
import time

from collections import namedtuple
from itertools import product
cwd = os.getcwd()
os.chdir("/mnt/nfs/scratch1/jpayan/openreview-matcher")
from matcher.solvers import FairSequence, FairFlow
os.chdir(cwd)

encoder = namedtuple(
    "Encoder", ["aggregate_score_matrix", "constraint_matrix"]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--algorithm", type=str)

    return parser.parse_args()


def get_valuation(paper, reviewer_set, paper_reviewer_affinities):
    val = 0
    for r in reviewer_set:
        val += paper_reviewer_affinities[r, paper]
    return val


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


def convert_soln_from_npy(npy_alloc):
    ffs = npy_alloc

    reviewers = list(range(ffs.shape[0]))
    papers = list(range(ffs.shape[1]))

    alloc = {p: set() for p in papers}

    for r in reviewers:
        for p in papers:
            if ffs[r, p] == 1:
                alloc[p].add(r)

    return alloc


if __name__ == "__main__":
    args = parse_args()
    dset = args.dataset
    data_dir = args.data_dir



    scores = np.load(os.path.join(data_dir, dset, "scores.npy"))
    loads = np.load(os.path.join(data_dir, dset, "loads.npy"))
    covs = np.load(os.path.join(data_dir, dset, "covs.npy"))
    constraint_matrix = np.zeros(scores.shape)
    mins = np.zeros(loads.shape)

    if args.algorithm == "FairFlow":
        solver_alg = FairFlow
    elif args.algorithm == "FairSequence":
        solver_alg = FairSequence

    runtimes = []
    usws = []
    nums_ef1_violations = []
    for _ in range(10):
        solver = solver_alg(
            mins,
            loads,
            covs,
            encoder(scores.transpose(), constraint_matrix.transpose()),
        )
        start = time.time()
        res = solver.solve()
        runtimes.append(time.time() - start)

        usw = np.sum(res.transpose() * scores)

        print(usw)
        print(usw / covs.shape[0])
        usws.append(usw / covs.shape[0])

        num_ef1_violations = ef1_violations(convert_soln_from_npy(res.transpose()), scores)
        print(num_ef1_violations)
        nums_ef1_violations.append(num_ef1_violations)

    with open("%s_%s_speed_test" % (dset, args.algorithm), 'w') as f:
        f.write("USW: %.2f pm %.2f, EF1 violations: %.2f pm %.2f, Runtime %.2f pm %.2f" %
                (np.mean(usws), np.std(usws),
                 np.mean(nums_ef1_violations), np.std(nums_ef1_violations),
                 np.mean(runtimes), np.std(runtimes)))