import random
import sys
import time

from utils import *


# Return the set of reviewers chosen before round t.
def get_alloc_up_to_rd(alloc, a, t):
    return {r for _t, r in alloc[a].items() if _t < t}


# This is a valid assignment if we are EF1 with respect to all agents earlier who have picked a worse reviewer than
# what we are trying to assign.
def is_valid_assignment(r, a, t, alloc, scores, previous_attained_scores):
    papers_to_check_against = set()
    a_alloc_after_t = get_alloc_up_to_rd(alloc, a, t) | {r}

    for rev in a_alloc_after_t:
        papers_to_check_against |= set(np.where(previous_attained_scores[t] < scores[rev, :])[0].tolist())

    for p in papers_to_check_against:
        p_alloc_after_t = get_alloc_up_to_rd(alloc, p, t)
        other = get_valuation(p, a_alloc_after_t, scores) - np.max(scores[a_alloc_after_t, [p] * len(a_alloc_after_t)])
        curr = get_valuation(p, p_alloc_after_t, scores)
        if other > curr:
            return False

    return True


# This is the tough one. We will need to look ahead and flag any papers that might need to repick (including self!).
# Then we have those
# guys repick, and they have to look ahead and flag papers, etc. Probably just make copies of everything before start.
# As I do this, keep track of a dict which maps papers to papers they've blocked from making selections due to EF1
# constraints.
def determine_mg(r, a, t, alloc, scores, loads_copy, previous_attained_scores, rounds, ef1_blocked_by):
    lc_copy = loads_copy.copy()
    pas_copy = previous_attained_scores.copy()
    a_copy = alloc.copy()
    k = len(rounds)

    a_copy[a][t] = r
    for _t in range(t, k):
        lc_copy[_t][r] -= 1
        pas_copy[_t][a] = min(pas_copy[_t][a], scores[r, a])

    fixup_queue = [(a, t)]

    while fixup_queue:
        a_fix, t_fix = fixup_queue.pop(0)
        r_fix = a_copy[a_fix][t_fix]

        # TODO: actually do the fix -> meaning, just pick a new choice
        # TODO: remember that we need to add the old choice back into the pool, and

        # TODO: Do the fix for anything ahead of me in the current round

        # Do the fix for anything in a future round
        if t_fix < len(rounds) - 1:
            for future_t in range(t_fix + 1, k):
                for future_a in rounds[future_t]:
                    if future_a in ef1_blocked_by[a_fix] or \
                            (lc_copy[t_fix][r_fix] == 0 and a_copy[future_a][future_t] == r_fix):
                        fixup_queue.append((future_a, future_t))

    # return everything, because if we go with assigning this reviewer to this agent, we will probably want to
    # just use the allocation, loads_copy, etc that we just computed.


def determine_next_selection(best_revs_map, loads_copy, available_agents, k,
                             scores, max_mg, alloc, previous_attained_scores, rounds):
    next_agent = [None] * k
    next_mg = [-1000] * k
    for t in range(k):
        for a in sorted(available_agents[t], key=lambda x: random.random()):
            if next_mg == max_mg:
                break
            for r in best_revs_map[t][a]:
                if loads_copy[t][r] <= 0 or r in get_alloc_up_to_rd(alloc, a, t):
                    best_revs_map[t][a].remove(r)
                if is_valid_assignment(r, a, t, alloc, scores, previous_attained_scores):
                    mg_for_a = determine_mg(r, a, t, alloc, scores, loads_copy, rounds)
                    if mg_for_a > next_mg[t]:
                        next_agent[t] = a
                        next_mg[t] = mg_for_a
                    break

    # next_round = np.argmax(next_mg)
    # next_agent_for_next_round = next_agent[next_round]
    # next_mg_for_next_round = next_mg[next_round]
    #
    # return next_round, next_agent_for_next_round, next_mg_for_next_round, \
    #        chosen_rev, new_alloc, best_revs_map, loads_copy


"""Create a recursively balanced picking sequence by greedily selecting the best remaining agent and round. 
Agents are selected to maximize the marginal gain."""


def grb(scores, loads, covs, best_revs, alloc_file, num_processes, sample_size):
    m, n = scores.shape

    k = covs[0]
    available_agents = [set(range(n))] * k
    rounds = [list()] * k
    alloc = {p: {} for p in available_agents}

    curr_usw = 0

    max_mg = np.max(scores)
    # previous_attained_scores holds the lowest affinity of any reviewer
    # chosen by each paper up to and including round t
    previous_attained_scores = [np.ones(n) * 1000] * k

    matrix_alloc = np.zeros((scores.shape), dtype=np.bool)

    loads_copy = loads.copy()

    best_revs_map = {}
    for t in range(k):
        best_revs_map[t] = {}
        for a in range(n):
            best_revs_map[t][a] = best_revs[:, a].tolist()

    # How to determine the marginal gain of an agent-round pair: call a recursive function "pick and check ahead".
    # This will let the agent pick its reviewer. Then it looks ahead and determines who might need to repick.
    # Who needs to repick: if we take the last copy of a reviewer, then we need to check ahead for anyone who picked
    # that reviewer. We also need to keep track of which papers rejected a reviewer because of EF1 violations with the
    # current paper.

    while len(ordering) < np.sum(covs):
        next_round, next_agent_for_next_round, next_mg_for_next_round, \
        chosen_rev, new_alloc, best_revs_map, loads_copy = determine_next_selection(best_revs_map,
                                                                                    loads_copy,
                                                                                    available_agents,
                                                                                    k,
                                                                                    scores,
                                                                                    max_mg,
                                                                                    alloc,
                                                                                    previous_attained_scores,
                                                                                    rounds)

        new_assn = False
        for r in best_revs_map[next_agent]:
            if loads_copy[r] > 0 and r not in alloc[next_agent]:
                loads_copy[r] -= 1
                alloc[next_agent].append(r)
                matrix_alloc[r, next_agent] = 1
                new_assn = True
                break

        print(next_agent)
        print(next_mg)
        print(len(ordering))
        if not new_assn:
            print("no new assn")
            return alloc, loads_copy, matrix_alloc

        ordering.append(next_agent)
        available_agents.remove(next_agent)

        if len(available_agents) == 0:
            round_num += 1
            available_agents = set(range(n))

    return alloc, ordering


def run_algo(dataset, base_dir, alloc_file, num_processes, sample_size):
    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)

    best_revs = np.argsort(-1 * scores, axis=0)

    alloc, ordering = grb(scores, loads, covs, best_revs, alloc_file, num_processes, sample_size)
    return alloc, ordering


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = args.base_dir
    alloc_file = args.alloc_file
    num_processes = args.num_processes
    sample_size = args.sample_size

    random.seed(args.seed)

    start = time.time()
    alloc, ordering = run_algo(dataset, base_dir, alloc_file, num_processes, sample_size)
    runtime = time.time() - start

    save_alloc(alloc, alloc_file)
    save_alloc(ordering, alloc_file + "_order")

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
