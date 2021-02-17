# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import argparse
import os

from autoassigner import *
from utils import *


def pr4a(pra, covs, loads):
    # Normalize the affinities so they're between 0 and 1
    pra[np.where(pra < 0)] = 0
    pra /= np.max(pra)

    start = time.time()

    pr4a_instance = auto_assigner(pra, demand=covs[0], ability=loads[0])
    pr4a_instance.fair_assignment()

    alg_time = time.time() - start

    alloc = pr4a_instance.fa

    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, covs, alg_time=alg_time)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="midl")
    parser.add_argument("--base_dir", type=str, default="/home/justinspayan/Fall_2020/fair-matching/data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    dataset = args.dataset

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)

    pr4a(paper_reviewer_affinities, covs, loads)