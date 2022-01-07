import random
import sys
import time

from copy import deepcopy
from utils import *


# Return the set of reviewers chosen before round t.
def get_alloc_up_to_rd(alloc, a, t):
    return {r for _t, r in alloc[a].items() if _t < t}


# This is a valid assignment if we are EF1 with respect to all agents earlier who have picked a worse reviewer than
# what we are trying to assign.
def is_valid_assignment(r, a, t, alloc, scores, previous_attained_scores):
    papers_to_check_against = set()
    a_alloc_after_t = list(get_alloc_up_to_rd(alloc, a, t) | {r})

    for rev in a_alloc_after_t:
        papers_to_check_against |= set(np.where(previous_attained_scores[t] < scores[rev, :])[0].tolist())

    for p in papers_to_check_against:
        p_alloc_after_t = list(get_alloc_up_to_rd(alloc, p, t))
        other = get_valuation(p, a_alloc_after_t, scores) - np.max(scores[a_alloc_after_t, [p] * len(a_alloc_after_t)])
        curr = get_valuation(p, p_alloc_after_t, scores)
        if other > curr:
            return False

    return True


# Just rerun the picking sequence in front of this guy.
def determine_mg(r, a, t, alloc, scores, loads_copy, previous_attained_scores, rounds, best_revs_map):
    # print("determine mg")
    # print(r)
    # print(a)
    # print(t)
    lc_copy = deepcopy(loads_copy)
    pas_copy = deepcopy(previous_attained_scores)
    a_copy = deepcopy(alloc)
    brm_copy = deepcopy(best_revs_map)
    k = len(rounds)

    rounds_copy = deepcopy(rounds)

    # print(rounds_copy)

    usw_before = 0
    for _t in range(k):
        for _a in rounds_copy[_t]:
            usw_before += scores[alloc[_a][_t], _a]

    rounds_copy[t].append(a)

    # print(usw_before)

    # print(rounds_copy)

    a_copy[a][t] = r

    # print([(i, j) for (i, j) in a_copy.items() if len(j)])

    for _t in range(t+1, k):
        lc_copy[_t] = lc_copy[t].copy()
        pas_copy[_t] = pas_copy[t].copy()

    for _t in range(t, k):
        lc_copy[_t][r] -= 1
        pas_copy[_t][a] = min(pas_copy[_t][a], scores[r, a])

    for _t in range(t + 1, k):
        for _a in rounds[_t]:
            for _r in brm_copy[_t][_a]:
                if lc_copy[_t][_r] <= 0 or _r in get_alloc_up_to_rd(a_copy, _a, _t):
                    brm_copy[_t][_a].remove(_r)
                elif is_valid_assignment(_r, _a, _t, a_copy, scores, pas_copy):
                    a_copy[_a][_t] = _r
                    lc_copy[_t][_r] -= 1
                    pas_copy[_t][_a] = min(pas_copy[_t][_a], scores[_r, _a])

    usw_after = 0
    for _t in range(k):
        for _a in rounds_copy[_t]:
            # print(_t, _a, a_copy[_a][_t])
            usw_after += scores[a_copy[_a][_t], _a]

    # print(usw_after)
    # sys.exit(0)
    return usw_after - usw_before, a_copy, lc_copy, pas_copy, brm_copy, rounds_copy


def determine_next_alloc(best_revs_map, loads_copy, available_agents, k,
                         scores, max_mg, alloc, previous_attained_scores, rounds):
    next_agent = [None] * k
    next_mg = [-1000] * k
    next_alloc = [None] * k
    next_lc = [None] * k
    next_pas = [None] * k
    next_brm = [None] * k
    next_rounds = [None] * k

    for t in range(k):
        for a in sorted(available_agents[t], key=lambda x: random.random()):
            if np.isclose(next_mg[t], max_mg):
                break
            for r in best_revs_map[t][a]:
                if loads_copy[t][r] <= 0 or r in get_alloc_up_to_rd(alloc, a, t):
                    for _t in range(t, k):
                        if r in best_revs_map[_t][a]:
                            best_revs_map[_t][a].remove(r)
                elif is_valid_assignment(r, a, t, alloc, scores, previous_attained_scores):
                    mg_for_a, \
                    tmp_new_alloc, \
                    tmp_new_lc, \
                    tmp_new_pas, \
                    tmp_new_brm, \
                    tmp_new_rounds = determine_mg(r, a, t, alloc, scores, loads_copy,
                                                  previous_attained_scores, rounds, best_revs_map)
                    if mg_for_a > next_mg[t]:
                        next_agent[t] = a
                        next_mg[t] = mg_for_a
                        next_alloc[t] = tmp_new_alloc
                        next_lc[t] = tmp_new_lc
                        next_pas[t] = tmp_new_pas
                        next_brm[t] = tmp_new_brm
                        next_rounds[t] = tmp_new_rounds
                    break

    print(next_agent)
    print(next_mg)
    print(next_alloc)

    best_round = np.argmax(next_mg)
    return next_alloc[best_round], next_lc[best_round], next_pas[best_round], \
           next_brm[best_round], next_rounds[best_round], next_agent[best_round], best_round


"""Create a recursively balanced picking sequence by greedily selecting the best remaining agent and round. 
Agents are selected to maximize the marginal gain."""


def grb(scores, loads, covs, best_revs, alloc_file, num_processes, sample_size):
    m, n = scores.shape

    k = covs[0]
    available_agents = [set(range(n))] * k
    rounds = [list() for _ in range(k)]
    alloc = {p: {} for p in available_agents[0]}

    curr_usw = 0

    max_mg = np.max(scores)
    # previous_attained_scores holds the lowest affinity of any reviewer
    # chosen by each paper up to and including round t
    previous_attained_scores = [np.ones(n) * 1000] * k
    loads_copy = [loads.copy()] * k

    best_revs_map = {}
    for t in range(k):
        best_revs_map[t] = {}
        for a in range(n):
            best_revs_map[t][a] = best_revs[:, a].tolist()

    # while sum([len(x) for x in rounds]) < np.sum(covs):
    for i in range(10):
        alloc, \
        loads_copy, \
        previous_attained_scores, \
        best_revs_map, \
        rounds,\
        next_agent,\
        next_round = determine_next_alloc(best_revs_map,
                                      loads_copy,
                                      available_agents,
                                      k,
                                      scores,
                                      max_mg,
                                      alloc,
                                      previous_attained_scores,
                                      rounds)
        available_agents[next_round].remove(next_agent)

    return alloc, rounds


def run_algo(dataset, base_dir, alloc_file, num_processes, sample_size):
    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)

    best_revs = np.argsort(-1 * scores, axis=0)

    alloc, rounds = grb(scores, loads, covs, best_revs, alloc_file, num_processes, sample_size)
    return alloc, rounds


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = args.base_dir
    alloc_file = args.alloc_file
    num_processes = args.num_processes
    sample_size = args.sample_size

    random.seed(args.seed)

    start = time.time()
    alloc, rounds = run_algo(dataset, base_dir, alloc_file, num_processes, sample_size)
    runtime = time.time() - start

    save_alloc(alloc, alloc_file)
    save_alloc(rounds, alloc_file + "_order")

    # print("Barman Algorithm Results")
    # print("%.2f seconds" % runtime)
    # print_stats(alloc, paper_reviewer_affinities, paper_capacities)

    print("Final Greedy Results")
    print("%.2f seconds" % runtime)

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy"))

    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, paper_capacities)
