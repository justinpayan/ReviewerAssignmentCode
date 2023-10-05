# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import re
import time

from gurobipy import *

from calculate_stats import *
from utils import *


def create_multidict(pra):
    d = {}
    for rev in range(pra.shape[0]):
        for paper in range(pra.shape[1]):
            d[(paper, rev)] = pra[rev, paper]
    return multidict(d)


def add_vars_to_model(m, paper_rev_pairs):
    x = m.addVars(paper_rev_pairs, name="assign", vtype=GRB.BINARY)  # The binary assignment variables
    return x


def add_constrs_to_model(m, x, covs, loads, cois):
    papers = range(covs.shape[0])
    revs = range(loads.shape[0])
    m.addConstrs((x.sum(paper, '*') == covs[paper] for paper in papers), 'covs')  # Paper coverage constraints
    m.addConstrs((x.sum('*', rev) <= loads[rev] for rev in revs), 'loads')  # Reviewer load constraints
    m.addConstrs(x <= 1-cois.T, 'cois')


def convert_to_dict(m, num_papers):
    alloc = {p: list() for p in range(num_papers)}
    for var in m.getVars():
        if var.varName.startswith("assign") and var.x > .1:
            s = re.findall("(\d+)", var.varName)
            p = int(s[0])
            r = int(s[1])
            alloc[p].append(r)
    return alloc


def tpms(pra, covs, loads, cois):
    # create a multidict which stores the paper reviewer affinities
    paper_rev_pairs, pras = create_multidict(pra)

    m = Model("TPMS")

    start = time.time()

    x = add_vars_to_model(m, paper_rev_pairs)
    add_constrs_to_model(m, x, covs, loads, cois)

    m.setObjective(x.prod(pras), GRB.MAXIMIZE)

    # m.write("TPMS.lp")

    m.optimize()

    alg_time = time.time() - start

    print("TPMS Score")
    print(m.objVal)

    # Convert to the format we were using, and then print it out and run print_stats
    alloc = convert_to_dict(m, covs.shape[0])

    return alloc


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    dataset = args.dataset
    alloc_file = args.alloc_file

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    cois = np.load(os.path.join(base_dir, dataset, "cois.npy"))

    alloc = tpms(paper_reviewer_affinities, covs, loads, cois)

    save_alloc(alloc, alloc_file)

    print(alloc)
    print_stats(alloc, paper_reviewer_affinities)