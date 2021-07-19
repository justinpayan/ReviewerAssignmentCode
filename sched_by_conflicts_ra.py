import os
import time

from utils import *


def schedule_by_conflicts(scores, loads, covs):
    # Convert scores to just duplicate the reviewers
    dups = loads[0]
    c = covs[0]

    matrix_alloc = np.zeros((scores.shape), dtype=np.bool)
    m, n = scores.shape

    # matrix_alloc = np.zeros((pra.shape), dtype=np.bool)
    #
    # loads_copy = loads.copy()
    #
    # # Assume all covs are the same
    # if output_alloc:
    #     for _ in range(covs[seln_order[0]]):
    #         for a in seln_order:
    #             for r in best_revs[:, a]:
    #                 if loads_copy[r] > 0 and r not in alloc[a]:
    #                     loads_copy[r] -= 1
    #                     alloc[a].append(r)
    #                     matrix_alloc[r, a] = 1
    #                     break
    #     return alloc, loads_copy, matrix_alloc

    best_revs = np.argsort(-1 * scores, axis=0)

    # remaining = np.ones((m)) * dups
    remaining = np.copy(loads)

    # Compute conflicts and ordering based on that
    # This works, it just blows up on my machine. There must be a way to reduce the precision and
    # get it to work, or maybe to split up the matrix multiplications
    use_accel = False
    items_in_order_all_agents = np.argsort(scores, axis=0)
    sorted_scores = np.sort(scores, axis=0)

    if use_accel:
        # sorted_per_agent = scores.astype(np.float16)[items_in_order_all_agents.transpose(), :]
        # sorted_scores_per_agent = np.tile(sorted_scores.astype(np.float16).transpose().reshape((n, 1, m)), (1, n, 1)).transpose(0, 2, 1)
        # conflict_map = sorted_per_agent >= sorted_scores_per_agent
        # conflicts = np.sum(conflict_map, axis=2).transpose() - 1
        conflicts = np.sum(scores[items_in_order_all_agents.transpose(), :]
                           >= np.tile(sorted_scores.transpose().reshape((n, 1, m)), (1, n, 1)).transpose((0, 2, 1)),
                           axis=2).transpose() - 1
        # conflict_map = sorted_per_agent >= sorted_scores_per_agent
        # conflicts = np.sum(conflict_map, axis=2).transpose() - 1
    else:
        conflicts = np.zeros(scores.shape)
        for a in range(n):
            if a % 50 == 0:
                print("getting conflicts: ", a)
            conflicts[:, a] = np.sum(scores[items_in_order_all_agents[:, a], :] >= sorted_scores[:, a].reshape((-1, 1)),
                                     axis=1)

    # conflicts = np.zeros(scores.shape)
    # for a in range(n):
    #     print("getting conflicts: ", a)
    #     items_in_order = items_in_order_all_agents[:, a]
    #     for pref_num in range(m):
    #         conflicts[m - pref_num - 1, a] = np.sum(scores[items_in_order[pref_num], :] >= (pref_num + 1)) - 1

    ordering = []
    for a in range(n):
        ordering.append((a, tuple(conflicts[:, a].tolist())))
    ordering = sorted(ordering, key=lambda x: x[1], reverse=True)
    ordering = [x[0] for x in ordering]

    print("ordering: ", ordering)

    for _ in range(c):
        for a in ordering:
            for r in best_revs[:, a]:
                if remaining[r] and matrix_alloc[r, a] < 1:
                    remaining[r] -= 1
                    matrix_alloc[r, a] = 1
                    break

    # print("in usw")
    print(matrix_alloc)
    # print(val_fns)
    alloc = {a: list(np.where(matrix_alloc[:, a])[0]) for a in range(n)}

    return alloc


def run_algo(dataset, base_dir):
    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)

    alloc = schedule_by_conflicts(paper_reviewer_affinities, reviewer_loads, paper_capacities)
    return alloc


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = args.base_dir
    alloc_file = args.alloc_file

    start = time.time()
    alloc = run_algo(dataset, base_dir)
    runtime = time.time() - start

    # alloc = load_alloc(alloc_file)
    save_alloc(alloc, alloc_file)

    # print("Barman Algorithm Results")
    # print("%.2f seconds" % runtime)
    # print_stats(alloc, paper_reviewer_affinities, paper_capacities)

    print("Schedule by Conflicts RR Results")
    print("%.2f seconds" % runtime)

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy"))

    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, paper_capacities)
