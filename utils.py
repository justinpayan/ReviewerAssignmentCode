import argparse
import numpy as np
import os
import pickle

from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="midl")
    parser.add_argument("--base_dir", type=str, default=os.path.join(os.getcwd(), "..", "fair-matching", "data"))
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--w_value", type=float, default=0.0)
    parser.add_argument("--alloc_file", type=str, default="allocation")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--sample_size", type=int, default=-1)
    parser.add_argument("--fair_matching_dir", type=str, default=os.path.join(os.getcwd(), "..", "fair-matching"))
    parser.add_argument("--fairflow_timestamp", type=str, default=None)
    parser.add_argument("--fairir_timestamp", type=str, default=None)
    return parser.parse_args()


def save_alloc(alloc, alloc_file):
    with open(alloc_file, 'wb') as f:
        pickle.dump(alloc, f)


def load_alloc(alloc_file):
    with open(alloc_file, 'rb') as f:
        return pickle.load(f)


# Return the usw of running round robin on the agents in the list "seln_order"
def safe_rr_usw(seln_order, pra, covs, loads, best_revs):
    # rr_alloc, rev_loads_remaining, matrix_alloc = rr(seln_order, pra, covs, loads, best_revs)
    _, rev_loads_remaining, matrix_alloc = safe_rr(seln_order, pra, covs, loads, best_revs)
    # _usw = usw(rr_alloc, pra)
    _usw = np.sum(matrix_alloc * pra)
    # print("USW ", time.time() - start)
    return _usw, rev_loads_remaining, matrix_alloc


def is_safe_choice_orderless(r, a, matrix_alloc, papers_who_tried_revs, pra, round_num, first_reviewer):
    if round_num == 0 or not len(papers_who_tried_revs[r]):
        return True

    # Construct the allocations we'll use for comparison
    a_alloc_orig = matrix_alloc[:, a]
    a_alloc_proposed = a_alloc_orig.copy()
    a_alloc_proposed[r] = 1
    a_alloc_proposed_reduced = a_alloc_proposed.copy()
    a_alloc_proposed_reduced[first_reviewer[a]] = 0

    for p in papers_who_tried_revs[r]:
        if p != a:
            # Check that they will not envy a if we add r to a.
            _a = a_alloc_proposed_reduced
            v_p_for_a_proposed = np.sum(_a * pra[:, p])

            v_p_for_p = np.sum(matrix_alloc[:, p] * pra[:, p])
            if v_p_for_a_proposed > v_p_for_p and not np.isclose(v_p_for_a_proposed, v_p_for_p):
                return False
    return True


"""We are trying to add r to a's bundle, but need to be sure we meet the criteria to prove the allocation is EF1. 
For all papers, we need to make sure the inductive step holds that a paper p always prefers its own assignment
to that of another paper (a, here) ~other than the 0th round with respect to p~. So we check, for any papers 
p earlier than a, do they prefer their own entire bundle to a's entire bundle? And for papers later than a, do they
prefer their own bundle to a's bundle other than the 1st reviewer? 

The first round is always safe.

Finally, we only need to check papers who have previously chosen this reviewer we are about to choose, because if 
they haven't chosen it they must have valued their own choices more. 
"""


def is_safe_choice(r, a, seln_order_idx_map, matrix_alloc, papers_who_tried_revs, pra, round_num, first_reviewer):
    if round_num == 0 or not len(papers_who_tried_revs[r]):
        return True
    a_idx = seln_order_idx_map[a]

    # Construct the allocations we'll use for comparison
    a_alloc_orig = matrix_alloc[:, a]
    a_alloc_proposed = a_alloc_orig.copy()
    a_alloc_proposed[r] = 1
    a_alloc_proposed_reduced = a_alloc_proposed.copy()
    a_alloc_proposed_reduced[first_reviewer[a]] = 0

    for p in papers_who_tried_revs[r]:
        if p != a:
            # Check that they will not envy a if we add r to a.
            _a = a_alloc_proposed if (seln_order_idx_map[p] < a_idx) else a_alloc_proposed_reduced
            v_p_for_a_proposed = np.sum(_a * pra[:, p])

            v_p_for_p = np.sum(matrix_alloc[:, p] * pra[:, p])
            if v_p_for_a_proposed > v_p_for_p and not np.isclose(v_p_for_a_proposed, v_p_for_p):
                return False
    return True


def safe_rr(seln_order, pra, covs, loads, best_revs):
    alloc = {p: list() for p in seln_order}
    matrix_alloc = np.zeros((pra.shape), dtype=np.bool)

    loads_copy = loads.copy()

    # When selecting a reviewer, you need to check for EF1 (inductive step) with all other papers who either
    # previously chose that reviewer or were themselves forced to pass over that reviewer... aka anyone
    # that ever TRIED to pick that reviewer. A paper that never
    # tried to pick that reviewer will have picked someone they liked better anyway.
    papers_who_tried_revs = defaultdict(list)
    first_reviewer = {}

    seln_order_idx_map = {p: idx for idx, p in enumerate(seln_order)}

    # Assume all covs are the same
    for round_num in range(covs[seln_order[0]]):
        for a in seln_order:
            new_assn = False
            for r in best_revs[:, a]:
                if loads_copy[r] > 0 and r not in alloc[a]:
                    if is_safe_choice(r, a, seln_order_idx_map, matrix_alloc,
                                      papers_who_tried_revs, pra, round_num, first_reviewer):
                        loads_copy[r] -= 1
                        alloc[a].append(r)
                        matrix_alloc[r, a] = 1
                        if round_num == 0:
                            first_reviewer[a] = r
                        papers_who_tried_revs[r].append(a)
                        new_assn = True
                        break
                    else:
                        papers_who_tried_revs[r].append(a)
            if not new_assn:
                print("no new assn")
                return alloc, loads_copy, matrix_alloc
    return alloc, loads_copy, matrix_alloc
