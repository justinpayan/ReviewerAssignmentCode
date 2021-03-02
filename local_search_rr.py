import math
import multiprocessing as mp
import os
import random
import time

from greedy_rr import rr_usw, rr, _greedy_rr_ordering
from tqdm import tqdm
from utils import *


class LocalSearcher(object):
    def __init__(self, scores, loads, covs, epsilon, initial_order, num_processes):
        self.best_revs = np.argsort(-1 * scores, axis=0)
        self.scores = scores
        self.loads = loads
        self.covs = covs
        self.m, self.n = scores.shape
        self.improvement_factor = 1 + epsilon / (self.n ** 4)
        self.initial_order = initial_order
        self.num_processes = num_processes

    @staticmethod
    def tuples_to_list(order):
        order = sorted(order, key=lambda x: x[1])
        return [t[0] for t in order]

    @staticmethod
    def list_to_tuples(order_list, ground_set, n):
        order = set()
        idx = 0
        for i in order_list:
            while (i, idx) not in ground_set:
                idx += 1
            if idx >= n:
                print("Error converting list ordering to tuples")
                raise Exception
            else:
                order.add((i, idx))
        return order

    def check_obj(self, order):
        # Just call the rr_usw function after turning the set-of-tuples order into a sorted-list-based order
        list_order = LocalSearcher.tuples_to_list(order)
        return rr_usw(list_order, self.scores, self.covs, self.loads, self.best_revs)

    def can_delete_or_exchange(self, assigned_agents, assigned_positions, order, matrix_alloc,
                               current_rev_loads, curr_usw, e):
        # Check if we can delete or add/exchange the tuple e to improve
        # update the reviewer loads when we delete or exchange
        if e[0] in assigned_agents and e[1] in assigned_positions:
            # Try a deletion operation
            new_order = order - {e}
            currently_assigned_revs = np.where(matrix_alloc[:, e[0]])
            # We can only gain benefit from deleting this paper/agent if it frees up a reviewer
            if np.any(current_rev_loads[currently_assigned_revs] == 0):
                new_usw, tmp_rev_loads, tmp_matrix_alloc = self.check_obj(new_order)
                if new_usw >= self.improvement_factor * curr_usw:
                    return True
        else:
            # Try an exchange operation
            del_1 = set()
            del_2 = set()

            if e[0] in assigned_agents or e[1] in assigned_positions:
                for f in order:
                    if f[0] == e[0]:
                        del_1.add(f)
                    elif f[1] == e[1]:
                        del_2.add(f)
            new_order = order - (del_1 | del_2)
            new_order.add(e)

            new_usw, tmp_rev_loads, tmp_matrix_alloc = self.check_obj(new_order)
            if new_usw >= self.improvement_factor * curr_usw:
                return True

        return False

    def local_search(self, ground_set, initial):
        order = initial
        curr_usw, current_rev_loads, matrix_alloc = self.check_obj(order)
        print("initial usw ", curr_usw)
        times_thru = 0

        all_agents = set(range(self.n))
        all_positions = set(range(self.n))

        assigned_agents = {a for (a, _) in order}
        assigned_positions = {p for (_, p) in order}

        ground_set = list(ground_set)

        can_improve = True
        while can_improve:
            can_improve = False

            # First just add any agents to any open positions that improve the usw by the required amount
            # If we check that the top reviewers for an agent aren't saturated first, we can just put the agent
            # anywhere free in the order and increment the USW without running round robin.
            print("additions")
            for a in all_agents - assigned_agents:
                chosen_revs = self.best_revs[:self.covs[a], a]
                usw_improvement = np.sum(self.scores[chosen_revs, a])
                if not np.any(current_rev_loads[chosen_revs] == 0) and \
                        curr_usw + usw_improvement >= self.improvement_factor * curr_usw:
                    # add this agent in a random position in the order
                    # update USW, assigned_agents, and assigned_positions (and current_rev_loads)
                    curr_usw += usw_improvement
                    position = random.choice(tuple(all_positions - assigned_positions))
                    assigned_agents.add(a)
                    assigned_positions.add(position)
                    order.add((a, position))
                    current_rev_loads[chosen_revs] -= 1
                    matrix_alloc[chosen_revs, a] = 1

            print("additions done")
            print("usw ", curr_usw)

            pool = mp.Pool(processes=self.num_processes)

            for idx in tqdm(range(math.ceil(len(ground_set) / self.num_processes))):
                elements_to_check = ground_set[
                                    idx * self.num_processes: min((idx + 1) * self.num_processes, len(ground_set))]
                print("checking elements:")
                print(elements_to_check)
                start = time.perf_counter()
                results = pool.map(lambda e: self.can_delete_or_exchange(assigned_agents,
                                                                         assigned_positions,
                                                                         order,
                                                                         matrix_alloc,
                                                                         current_rev_loads,
                                                                         curr_usw,
                                                                         e),
                                   elements_to_check)
                successes = np.array(list(results))
                print(time.perf_counter() - start)
                print(np.any(successes))
                # Run exactly 1 of them synchronously. Of course, we could run all the successful ones synchronously
                # but we don't know if some of them will stop being useful once we run the first. So we'll just
                # ignore all but the first and if it's still relevant by the time we circle back around then great.
                if np.any(successes):
                    print("improving the ordering")
                    start = time.perf_counter()

                    e = elements_to_check[np.where(successes)[0][0]]

                    if e[0] in assigned_agents and e[1] in assigned_positions:
                        # Try a deletion operation
                        new_order = order - {e}
                        currently_assigned_revs = np.where(matrix_alloc[:, e[0]])
                        # We can only gain benefit from deleting this paper/agent if it frees up a reviewer
                        if np.any(current_rev_loads[currently_assigned_revs] == 0):
                            new_usw, tmp_rev_loads, tmp_matrix_alloc = self.check_obj(new_order)
                            if new_usw >= self.improvement_factor * curr_usw:
                                can_improve = True
                                curr_usw = new_usw
                                order = new_order
                                assigned_agents -= {e[0]}
                                assigned_positions -= {e[1]}
                                current_rev_loads = tmp_rev_loads
                                matrix_alloc = tmp_matrix_alloc
                    else:
                        # Try an exchange operation
                        del_1 = set()
                        del_2 = set()

                        if e[0] in assigned_agents or e[1] in assigned_positions:
                            for f in order:
                                if f[0] == e[0]:
                                    del_1.add(f)
                                elif f[1] == e[1]:
                                    del_2.add(f)
                        new_order = order - (del_1 | del_2)
                        new_order.add(e)

                        new_usw, tmp_rev_loads, tmp_matrix_alloc = self.check_obj(new_order)
                        if new_usw >= self.improvement_factor * curr_usw:
                            can_improve = True
                            curr_usw = new_usw
                            order = new_order
                            assigned_agents -= {t[0] for t in (del_1 | del_2)}
                            assigned_agents.add(e[0])
                            assigned_positions -= {t[1] for t in (del_1 | del_2)}
                            assigned_positions.add(e[1])
                            current_rev_loads = tmp_rev_loads
                            matrix_alloc = tmp_matrix_alloc
                    print(time.perf_counter() - start)

                    # THE BELOW WAS AN INTENDED SHORTCUT WHERE WE FIRST CHECK IF WE CAN JUST COMPUTE THE NEW USW
                    # ADDITIVELY INSTEAD OF RUNNING RR. IT RUNS A SMALL FRACTION OF THE TIME, SO WE CAN ABANDON THIS PLAN
                    # If the deletions do not free up a new reviewer and the addition does not put a reviewer at the
                    # limit, we can compute the new USW additively. On the other hand, if the deletions free up a reviewer
                    # or the added agent wants a reviewer at the limit, we will need to run round robin.
                    # del_1_freeing = False
                    # del_2_freeing = False
                    # if del_1:
                    #     currently_assigned_revs_1 = np.where(matrix_alloc[:, list(del_1)[0][0]])
                    #     del_1_freeing = np.any(current_rev_loads[currently_assigned_revs_1] == 0)
                    # if del_2:
                    #     currently_assigned_revs_2 = np.where(matrix_alloc[:, list(del_2)[0][0]])
                    #     del_2_freeing = np.any(current_rev_loads[currently_assigned_revs_2] == 0)
                    #
                    # chosen_revs = self.best_revs[:self.covs[e[0]], e[0]]
                    # addition_limiting = np.any(current_rev_loads[chosen_revs] == 0)
                    #
                    #
                    # if del_1_freeing or del_2_freeing or addition_limiting:
                    #     # print("rr")
                    #     new_usw, tmp_rev_loads, tmp_matrix_alloc = self.check_obj(new_order)
                    #     if new_usw >= self.improvement_factor * curr_usw:
                    #         print("success!!!!!!!!!")
                    #         print(new_usw)
                    #         can_improve = True
                    #         curr_usw = new_usw
                    #         order = new_order
                    #         assigned_agents -= {t[0] for t in (del_1 | del_2)}
                    #         assigned_agents.add(e[0])
                    #         assigned_positions -= {t[1] for t in (del_1 | del_2)}
                    #         assigned_positions.add(e[1])
                    #         current_rev_loads = tmp_rev_loads
                    #         matrix_alloc = tmp_matrix_alloc
                    #     else:
                    #         print("fail")
                    # else:
                    #     # print("shortcut")
                    #     usw_improvement = np.sum(self.scores[chosen_revs, e[0]])
                    #     usw_drop = 0
                    #
                    #     tmp_rev_loads = current_rev_loads.copy()
                    #     tmp_rev_loads[chosen_revs] -= 1
                    #     tmp_matrix_alloc = matrix_alloc.copy()
                    #     tmp_matrix_alloc[chosen_revs, e[0]] = 1
                    #
                    #     if del_1:
                    #         del_agent_1 = list(del_1)[0][0]
                    #         currently_assigned_revs_1 = np.where(matrix_alloc[:, del_agent_1])
                    #         usw_drop += np.sum(self.scores[currently_assigned_revs_1, del_agent_1])
                    #         tmp_rev_loads[currently_assigned_revs_1] += 1
                    #         tmp_matrix_alloc[currently_assigned_revs_1, del_agent_1] = 0
                    #     if del_2:
                    #         del_agent_2 = list(del_2)[0][0]
                    #         currently_assigned_revs_2 = np.where(matrix_alloc[:, del_agent_2])
                    #         usw_drop += np.sum(self.scores[currently_assigned_revs_2, del_agent_2])
                    #         tmp_rev_loads[currently_assigned_revs_2] += 1
                    #         tmp_matrix_alloc[currently_assigned_revs_2, del_agent_2] = 0
                    #
                    #     new_usw = curr_usw + usw_improvement - usw_drop
                    #     if new_usw >= self.improvement_factor * curr_usw:
                    #         can_improve = True
                    #         curr_usw = new_usw
                    #         order = new_order
                    #         assigned_agents -= {t[0] for t in (del_1 | del_2)}
                    #         assigned_agents.add(e[0])
                    #         assigned_positions -= {t[1] for t in (del_1 | del_2)}
                    #         assigned_positions.add(e[1])
                    #         current_rev_loads = tmp_rev_loads
                    #         matrix_alloc = tmp_matrix_alloc
                    # END OF THE SHORTCUT

            print(times_thru)
            print(order)
            print(curr_usw)
            times_thru += 1
        return order, curr_usw

    def get_approx_best_rr(self):
        ground_set = set(product(range(self.n), range(self.n)))

        # Run the heuristic to get a pretty decent partial allocation. This is a list,
        # but we can convert to a set of tuples depending on the ground set each of the 3 times we run local search.
        if not self.initial_order:
            initial_ordering = _greedy_rr_ordering(self.scores, self.covs, self.loads)
        else:
            initial_ordering = self.initial_order

        # Run the algorithm from Lee et al. 2009 to get a 4+epsilon approximation to the best RR allocation, for
        # a subset of the agents
        rr_orders = []
        for _ in range(3):
            initial_ordering = LocalSearcher.list_to_tuples(initial_ordering, ground_set, self.n)
            ls = self.local_search(ground_set, initial=initial_ordering)
            print("\n\ndone\n\n")
            rr_orders.append(ls)
            ground_set -= ls[0]

        # Pick the best order, compute that partial allocation
        best_option = sorted(rr_orders, key=lambda x: x[1])[-1][0]
        # Convert to list, use the other file's functions to get alloc
        partial_order = LocalSearcher.tuples_to_list(best_option)
        partial_alloc, _, _ = rr(partial_order, self.scores, self.covs, self.loads, self.best_revs, output_alloc=True)

        # TODO: run Lipton to complete the allocation. Can we guarantee this will always work?

        return partial_alloc


def run_algo(dataset, base_dir, epsilon, initial_order, num_processes):
    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)

    if initial_order:
        with open(initial_order, "rb") as f:
            initial_order = pickle.load(f)

    local_searcher = LocalSearcher(paper_reviewer_affinities, reviewer_loads, paper_capacities, epsilon, initial_order,
                                   num_processes)

    alloc = local_searcher.get_approx_best_rr()
    return alloc


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = args.base_dir
    alloc_file = args.alloc_file
    initial_order = args.local_search_init_order
    num_processes = args.num_processes

    random.seed(args.seed)

    epsilon = 1 / 5
    start = time.time()
    alloc = run_algo(dataset, base_dir, epsilon, initial_order, num_processes)
    runtime = time.time() - start

    save_alloc(alloc, alloc_file)

    # print("Barman Algorithm Results")
    # print("%.2f seconds" % runtime)
    # print_stats(alloc, paper_reviewer_affinities, paper_capacities)

    print("Local Search RR Results")
    print("%.2f seconds" % runtime)

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy"))

    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, paper_capacities)
