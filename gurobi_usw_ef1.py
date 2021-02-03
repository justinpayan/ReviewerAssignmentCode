# Produce an allocation which has maximal USW subject to the EF1 constraint
# We will just encode the EF1 constraint in Gurobi and see what happens.

import argparse
import os
import re

from gurobipy import *
import gurobipy as gp

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


def add_ef1_constraints(m, x, pras):
    papers = range(covs.shape[0])
    revs = range(loads.shape[0])
    # Add the V_1, V_2, etc variables, and constrain them to be equal to the current valuation for that paper
    print("adding vals")
    v_p = m.addVars(len(papers), name="V")
    m.addConstrs((v_p[p] == gp.quicksum(x[p, r] * pras[p, r] for r in revs) for p in papers), 'vals')

    # Add the V_1_2_1, V_1_2_2, etc variables. Set them equal to what they are supposed to equal
    valued_revs = {}
    for p in papers:
        valued_revs[p] = set()
        for rev in revs:
            if pras[p, rev] > 0:
                valued_revs[p].add(rev)

    d = []
    for p1 in papers:
        for p2 in papers:
            if p1 != p2 and (valued_revs[p1] & valued_revs[p2]):
                d.append((p1, p2))

    pairs_and_revs = []
    for p1 in papers:
        for p2 in papers:
            if p1 != p2 and (valued_revs[p1] & valued_revs[p2]):
                for rev in (valued_revs[p1] & valued_revs[p2]):
                    pairs_and_revs.append((p1, p2, rev))

    print(len(pairs_and_revs))
    print("adding cross_vals")
    v_p1_p2_r = m.addVars(pairs_and_revs, name="V_p1_p2_r")
    # Only count reviewers that have value for p[1] when computing initial value, since revs with no value
    # shouldn't be assigned anyway.
    # Also, only consider constraints for reviewers that hold value for both.
    m.addConstrs((v_p1_p2_r[p[0], p[1], p[2]] == gp.quicksum(x[p[1], r] * pras[p[0], r]
                                                             for r in valued_revs[p[1]]) -
                  x[p[1], p[2]] * pras[p[0], p[2]]
                  for p in pairs_and_revs), 'cross_vals')

    print("adding min_cross_ef1_vals")
    # Create variables V_1_1, V_1_2, etc, set them equal to the min over r of V_i_j_r
    v_p1_p2 = m.addVars(d, name="V_p1_p2")
    m.addConstrs((v_p1_p2[p[0], p[1]] == min_((v_p1_p2_r[p[0], p[1], r]
                                               for r in (valued_revs[p[0]] & valued_revs[p[1]]))) for p in d),
                 'min_cross_ef1_vals')

    # d = []
    # for p1 in papers:
    #     for p2 in papers:
    #         if p1 != p2:
    #             d.append((p1, p2))
    #
    # pairs_and_revs = []
    # for p1 in papers:
    #     for p2 in papers:
    #         if p1 != p2:
    #             for rev in revs:
    #                 pairs_and_revs.append((p1, p2, rev))
    #
    # print("adding cross_vals")
    # v_p1_p2_r = m.addVars(pairs_and_revs, name="V_p1_p2_r")
    # m.addConstrs((v_p1_p2_r[p[0], p[1], rev] == gp.quicksum(x[p[1], r] * pras[p[0], r] for r in revs) -
    #               x[p[1], rev] * pras[p[0], rev]
    #               for rev in revs
    #               for p in d), 'cross_vals')
    #
    # print("adding min_cross_ef1_vals")
    # # Create variables V_1_1, V_1_2, etc, set them equal to the min over r of V_i_j_r
    # v_p1_p2 = m.addVars(d, name="V_p1_p2")
    # m.addConstrs((v_p1_p2[p[0], p[1]] == min_((v_p1_p2_r[p[0], p[1], r] for r in revs)) for p in d), 'min_cross_ef1_vals')

    print("adding ef1")
    # Require V_i >= V_i_j for all j.
    m.addConstrs((v_p[p[0]] >= v_p1_p2[p[0], p[1]] for p in d), 'ef1')


def convert_to_dict(m, num_papers):
    alloc = {p: list() for p in range(num_papers)}
    for var in m.getVars():
        if var.varName.startswith("assign") and var.x > .1:
            s = re.findall("(\d+)", var.varName)
            p = int(s[0])
            r = int(s[1])
            alloc[p].append(r)
    return alloc


def tpms(pra, covs, loads):
    # create a multidict which stores the paper reviewer affinities
    paper_rev_pairs, pras = create_multidict(pra)

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


def usw_ef1(pra, covs, loads):
    # create a multidict which stores the paper reviewer affinities
    paper_rev_pairs, pras = create_multidict(pra)

    m = Model("EF1")

    x = add_vars_to_model(m, paper_rev_pairs)
    add_constrs_to_model(m, x, covs, loads)

    # Add the additional variables and constraints
    add_ef1_constraints(m, x, pras)

    m.setObjective(x.prod(pras), GRB.MAXIMIZE)

    m.write("EF1.lp")

    m.optimize()

    print("EF1 USW")
    print(m.objVal)

    # Convert to the format we were using, and then print it out and run print_stats
    alloc = convert_to_dict(m, covs.shape[0])

    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, covs)


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

    tpms(paper_reviewer_affinities, covs, loads)

    usw_ef1(paper_reviewer_affinities, covs, loads)
