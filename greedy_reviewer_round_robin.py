
import time
import multiprocessing as mp
import random

from calculate_stats import *
from utils import *


def compute_usw(args):
    ordering, scores, covs, loads, best_revs = args
    return safe_rr_usw(ordering, scores, covs, loads, best_revs)[0]


"""Approximately compute the set of tuples corresponding to a total ordering which maximizes the USW from Reviewer RR,
by running the greedy algorithm. This greedy algorithm will always put an agent in the next available position in
the ordering, selecting the agent which maximizes the utilitarian welfare of the suborder at each step."""


def greedy(scores, loads, covs, best_revs, alloc_file, num_processes, sample_size):
    m, n = scores.shape

    available_agents = set(range(n))
    ordering = []
    alloc = None

    curr_usw = 0
    max_mg = covs[0]*np.max(scores)

    pool = mp.Pool(processes=num_processes)

    while len(ordering) < n:
        print("%0.2f percent finished. USW is %0.2f" % (100*(len(ordering)/n), curr_usw), flush=True)
        next_agent = None
        best_usw = -1

        if len(available_agents) < 20:
            for a in sorted(available_agents, key=lambda x: random.random()):
                usw = safe_rr_usw(ordering + [a], scores, covs, loads, best_revs)[0]
                if usw > best_usw:
                    best_usw = usw
                    next_agent = a
                if best_usw - curr_usw == max_mg:
                    break
        else:
            sorted_agents = sorted(available_agents)
            if 0 < sample_size < len(sorted_agents):
                sorted_agents = sorted(random.sample(sorted_agents, sample_size))
            all_orderings = [ordering + [a] for a in sorted_agents]
            list_of_copied_args = [all_orderings]
            for argument in [scores, covs, loads, best_revs]:
                list_of_copied_args.append(len(all_orderings) * [argument])

            usws = pool.map(compute_usw, zip(*list_of_copied_args))

            for a, usw in zip(sorted_agents, usws):
                if usw > best_usw:
                    best_usw = usw
                    next_agent = a
                if best_usw - curr_usw == max_mg:
                    break

        if len(ordering) % 100 == 1:
            print("Saving at len(ordering) %d" % len(ordering))
            save_alloc(alloc, alloc_file)
            save_alloc(ordering, alloc_file + "_order")

        curr_usw = best_usw
        ordering.append(next_agent)
        available_agents.remove(next_agent)

    alloc = safe_rr(ordering, scores, covs, loads, best_revs)[0]

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
    print_stats(alloc, paper_reviewer_affinities)
