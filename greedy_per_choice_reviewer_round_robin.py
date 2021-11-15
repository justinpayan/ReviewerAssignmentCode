import time
import multiprocessing as mp
import sys

from utils import *


def compute_usw(args):
    ordering, scores, covs, loads, best_revs = args
    return safe_rr_usw(ordering, scores, covs, loads, best_revs)[0]


"""Create a recursively balanced picking sequence by greedily selecting the best remaining agent at each step in each
round. Agents are selected to maximize the marginal gain."""


def greedy(scores, loads, covs, best_revs, alloc_file, num_processes, sample_size):
    m, n = scores.shape

    available_agents = set(range(n))
    ordering = []
    alloc = {p: list() for p in available_agents}

    curr_usw = 0

    round_num = 0

    max_mg = np.max(scores)

    matrix_alloc = np.zeros((scores.shape), dtype=np.bool)

    loads_copy = loads.copy()

    best_revs_map = {}
    for a in range(n):
        best_revs_map[a] = best_revs[:, a].tolist()

    # Maintain the invariant that no paper can take a reviewer that is worth more to some other paper
    # than what that paper has chosen in previous rounds. So basically, each round we will construct
    # the vector of the realized values for all papers. Then when you try to select a reviewer, you check
    # if np.any(scores[r, :] > previous_attained_scores). If so, you move on. Actually, it doesn't need to be
    # per round. Suppose that i comes before j in round t. Then suppose i picks something, and my non-per-round
    # update rules out what j was going to pick (because it is worth more to i). If j is allowed to pick this thing, i
    # would have been ok to pick it too. But it didn't.

    previous_attained_scores = np.ones(n) * 1000

    max_mg_per_agent = np.ones(n) * 1000

    while len(ordering) < np.sum(covs):
        for r in range(m):
            if np.any(scores[r, :] > previous_attained_scores):
                # Remove it
                loads_copy[r] = 0

        next_agent = None
        next_reviewer = None
        next_mg = -10000
        for a in sorted(available_agents, key=lambda x: random.random()):
            if max_mg_per_agent[a] >= next_mg:
                if next_mg == max_mg:
                    break
                for r in best_revs_map[a]:
                    if loads_copy[r] <= 0 or r in alloc[a]:
                        best_revs_map[a].remove(r)
                    else:
                        max_mg_per_agent[a] = scores[r, a]
                    if scores[r, a] > next_mg:
                        if loads_copy[r] > 0 and \
                                r not in alloc[a]:
                            next_agent = a
                            next_mg = scores[r, a]
                            break
                    else:
                        break

        new_assn = False
        for r in best_revs_map[next_agent]:
            if loads_copy[r] > 0 and r not in alloc[next_agent]:
                if np.any(previous_attained_scores < scores[r, :]):
                    print("error")
                    sys.exit(0)
                loads_copy[r] -= 1
                alloc[next_agent].append(r)
                matrix_alloc[r, next_agent] = 1
                previous_attained_scores[next_agent] = scores[r, next_agent]
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

    alloc, ordering = greedy(scores, loads, covs, best_revs, alloc_file, num_processes, sample_size)
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
