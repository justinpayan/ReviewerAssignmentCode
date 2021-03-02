import random
import time

from greedy_rr import rr_usw, rr
from itertools import product
from utils import *


class LocalSearcher(object):
    def __init__(self, scores, loads, covs, epsilon):
        self.best_revs = np.argsort(-1*scores, axis=0)
        self.scores = scores
        self.loads = loads
        self.covs = covs
        self.m, self.n = scores.shape
        self.improvement_factor = 1 + epsilon / (self.n ** 4)

    @staticmethod
    def tuples_to_list(order):
        order = sorted(order, key=lambda x: x[1])
        return [t[0] for t in order]

    def check_obj(self, order):
        # Just call the rr_usw function after turning the set-of-tuples order into a sorted-list-based order
        list_order = LocalSearcher.tuples_to_list(order)
        return rr_usw(list_order, self.scores, self.covs, self.loads, self.best_revs)

    def local_search(self, ground_set, initial=None):
        order = set()
        curr_usw = -1
        times_thru = 0

        all_agents = set(range(self.n))
        all_positions = set(range(self.n))

        assigned_agents = set()
        assigned_positions = set()

        current_rev_loads = self.loads.copy()
        matrix_alloc = np.zeros((self.scores.shape))

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
                    position = random.choice(tuple(all_positions-assigned_positions))
                    assigned_agents.add(a)
                    assigned_positions.add(position)
                    order.add((a, position))
                    current_rev_loads[chosen_revs] -= 1
                    matrix_alloc[chosen_revs, a] = 1

            print("additions done")
            print("usw ", curr_usw)

            # Check if we can delete or add/exchange a tuple to improve
            # update the reviewer loads when we delete or exchange
            idx = 0
            for e in ground_set:
                if idx % 100 == 0:
                    print("%.5f" % (float(idx)/len(ground_set)))
                idx += 1
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
                    # If the deletions do not free up a new reviewer and the addition does not put a reviewer at the
                    # limit, we can compute the new USW additively. On the other hand, if the deletions free up a reviewer
                    # or the added agent wants a reviewer at the limit, we will need to run round robin.
                    del_1_freeing = False
                    del_2_freeing = False
                    if del_1:
                        currently_assigned_revs_1 = np.where(matrix_alloc[:, list(del_1)[0][0]])
                        del_1_freeing = np.any(current_rev_loads[currently_assigned_revs_1] == 0)
                    if del_2:
                        currently_assigned_revs_2 = np.where(matrix_alloc[:, list(del_2)[0][0]])
                        del_2_freeing = np.any(current_rev_loads[currently_assigned_revs_2] == 0)

                    chosen_revs = self.best_revs[:self.covs[e[0]], e[0]]
                    addition_limiting = np.any(current_rev_loads[chosen_revs] == 0)


                    if del_1_freeing or del_2_freeing or addition_limiting:
                        # print("rr")
                        new_usw, tmp_rev_loads, tmp_matrix_alloc = self.check_obj(new_order)
                        if new_usw >= self.improvement_factor * curr_usw:
                            print("success!!!!!!!!!")
                            print(new_usw)
                            can_improve = True
                            curr_usw = new_usw
                            order = new_order
                            assigned_agents -= {t[0] for t in (del_1 | del_2)}
                            assigned_agents.add(e[0])
                            assigned_positions -= {t[1] for t in (del_1 | del_2)}
                            assigned_positions.add(e[1])
                            current_rev_loads = tmp_rev_loads
                            matrix_alloc = tmp_matrix_alloc
                        else:
                            print("fail")
                    else:
                        # print("shortcut")
                        usw_improvement = np.sum(self.scores[chosen_revs, e[0]])
                        usw_drop = 0

                        tmp_rev_loads = current_rev_loads.copy()
                        tmp_rev_loads[chosen_revs] -= 1
                        tmp_matrix_alloc = matrix_alloc.copy()
                        tmp_matrix_alloc[chosen_revs, e[0]] = 1

                        if del_1:
                            del_agent_1 = list(del_1)[0][0]
                            currently_assigned_revs_1 = np.where(matrix_alloc[:, del_agent_1])
                            usw_drop += np.sum(self.scores[currently_assigned_revs_1, del_agent_1])
                            tmp_rev_loads[currently_assigned_revs_1] += 1
                            tmp_matrix_alloc[currently_assigned_revs_1, del_agent_1] = 0
                        if del_2:
                            del_agent_2 = list(del_2)[0][0]
                            currently_assigned_revs_2 = np.where(matrix_alloc[:, del_agent_2])
                            usw_drop += np.sum(self.scores[currently_assigned_revs_2, del_agent_2])
                            tmp_rev_loads[currently_assigned_revs_2] += 1
                            tmp_matrix_alloc[currently_assigned_revs_2, del_agent_2] = 0

                        new_usw = curr_usw + usw_improvement - usw_drop
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

            print(times_thru)
            print(order)
            print(curr_usw)
            times_thru += 1
        return order, curr_usw

    def get_approx_best_rr(self):
        ground_set = set(product(range(self.n), range(self.n)))

        # First run the algorithm from Lee et al. 2009 to get a 4+epsilon approximation to the best RR allocation, for
        # a subset of the agents
        rr_orders = []
        for _ in range(3):
            ls = self.local_search(ground_set, initial=)
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


def run_algo(dataset, epsilon):
    paper_reviewer_affinities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/scores.npy" % dataset)
    reviewer_loads = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/loads.npy" % dataset).astype(np.int64)
    paper_capacities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/covs.npy" % dataset).astype(np.int64)

    local_searcher = LocalSearcher(paper_reviewer_affinities, reviewer_loads, paper_capacities, epsilon)

    alloc = local_searcher.get_approx_best_rr()
    return alloc


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    alloc_file = args.alloc_file

    random.seed(args.seed)

    epsilon = 1/5
    start = time.time()
    alloc = run_algo(dataset, epsilon)
    runtime = time.time() - start

    save_alloc(alloc, alloc_file)

    # print("Barman Algorithm Results")
    # print("%.2f seconds" % runtime)
    # print_stats(alloc, paper_reviewer_affinities, paper_capacities)

    print("Local Search RR Results")
    print("%.2f seconds" % runtime)
    paper_reviewer_affinities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/scores.npy" % dataset)
    paper_capacities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/covs.npy" % dataset)
    reviewer_loads = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/loads.npy" % dataset).astype(np.int64)
    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, paper_capacities)



