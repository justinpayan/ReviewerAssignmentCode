# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import argparse
import os
import re

from gurobipy import *

from barman_item_limits import *


def create_multidict(pra):
    d = {}
    for rev in range(pra.shape[0]):
        for paper in range(pra.shape[1]):
            d[(paper, rev)] = pra[rev, paper]
    return multidict(d)


def add_vars_to_model(m, paper_rev_pairs):
    x = m.addVars(paper_rev_pairs, name="assign", vtype=GRB.BINARY)  # The binary assignment variables
    return x


def add_constrs_to_model(m, x, covs, loads):
    papers = range(covs.shape[0])
    revs = range(loads.shape[0])
    m.addConstrs((x.sum(paper, '*') == covs[paper] for paper in papers), 'covs')  # Paper coverage constraints
    m.addConstrs((x.sum('*', rev) <= loads[rev] for rev in revs), 'loads')  # Reviewer load constraints


def convert_to_dict(m, num_papers):
    alloc = {p: list() for p in range(num_papers)}
    for var in m.getVars():
        if var.varName.startswith("assign") and var.x > .1:
            s = re.findall("(\d+)", var.varName)
            p = int(s[0])
            r = int(s[1])
            alloc[p].append(r)
    return alloc


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

    # create a multidict which stores the paper reviewer affinities
    paper_rev_pairs, pras = create_multidict(paper_reviewer_affinities)

    m = Model("TPMS")

    x = add_vars_to_model(m, paper_rev_pairs)
    add_constrs_to_model(m, x, covs, loads)

    m.setObjective(x.prod(pras), GRB.MAXIMIZE)

    m.write("TPMS.lp")

    m.optimize()

    print("TPMS Score")
    print(m.objVal)

    # Convert to the format we were using, and then print it out and run print_stats
    alloc = convert_to_dict(m, covs.shape[0])

    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, covs)
