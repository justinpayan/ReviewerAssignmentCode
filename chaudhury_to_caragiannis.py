import math
import networkx as nx

from usw_and_ef1 import *
from copy import deepcopy

def construct_graph(reviewer_loads, paper_reviewer_affinities, paper_capacities):
    # 1d numpy array, 2d array, 1d numpy array
    graph = nx.DiGraph()

    reviewers = list(range(reviewer_loads.shape[0]))
    reviewers = [i + 2 for i in reviewers]
    papers = list(range(paper_capacities.shape[0]))
    papers = [i + len(reviewers) + 2 for i in papers]

    supply_and_demand = np.sum(paper_capacities)
    graph.add_node(0, demand=int(-1*supply_and_demand))
    graph.add_node(1, demand=int(supply_and_demand))

    for r in reviewers:
        graph.add_node(r, demand=0)
    for p in papers:
        graph.add_node(p, demand=0)

    # Draw edges from reviewers to papers
    W = int(1e10)
    for p in papers:
        for r in reviewers:
            graph.add_edge(r, p, weight=-1*int(W*paper_reviewer_affinities[r-2, p-len(reviewers)-2]), capacity=1)
    # Draw edges from source to reviewers
    for r in reviewers:
        graph.add_edge(0, r, weight=0, capacity=int(reviewer_loads[r-2]))
    # Draw edges from papers to sink
    for p in papers:
        graph.add_edge(p, 1, weight=0, capacity=int(paper_capacities[p-len(reviewers)-2]))

    return graph


def get_alloc_from_flow_result(flowDict, loads, covs):
    reviewers = list(range(loads.shape[0]))
    reviewers = [i + 2 for i in reviewers]
    papers = list(range(covs.shape[0]))
    papers = [i + len(reviewers) + 2 for i in papers]

    alloc = {p - len(reviewers) - 2: set() for p in papers}

    for r in reviewers:
        for p in papers:
            if flowDict[r][p] == 1:
                alloc[p - len(reviewers) - 2].add(r - 2)

    return alloc


def perform_rounding(pra, pc, r):
    pra[np.where(pra > 0)] = r**np.ceil(np.log(pra[np.where(pra > 0)])/np.log(r))
    pc[np.where(pc > 0)] = r**np.ceil(np.log(pc[np.where(pc > 0)])/np.log(r))
    return pra, pc


def greedy_initial_assignment(loads, scores):
    m, n = scores.shape

    alloc = {p: list() for p in range(n)}
    prices = {r: 0.0 for r in range(m)}

    for rev in range(m):
        top_papers = np.argsort(scores[rev])[::-1][:loads[rev]]
        min_score = np.inf
        for p in top_papers:
            if scores[rev, p]:
                alloc[p].append(rev)
                min_score = min(min_score, scores[rev, p])
        prices[rev] = min_score

    return alloc, prices


def eps_p_ef1(alloc, epsilon, scores=None, alpha=None, prices=None):
    lowers = []
    uppers = []

    for paper in alloc:
        if alpha is not None and scores is not None:
            this_paper_prices = [scores[rev, paper]/alpha[paper] for rev in alloc[paper]]
        elif prices is not None:
            this_paper_prices = [prices[rev] for rev in alloc[paper]]

        if len(this_paper_prices) > 0:
            sum_prices = sum(this_paper_prices)
            lowers.append(sum_prices - max(this_paper_prices))
            uppers.append((1+epsilon) * sum_prices)
        else:
            lowers.append(0)
            uppers.append(0)

    return max(lowers) <= min(uppers)


def compute_p(paper, alloc, scores, alpha):
    return sum([scores[rev, paper] for rev in alloc[paper]])/alpha[paper]


def select_i(alloc, prices, caps, scores, alpha, disallowed_i):
    i = None
    i_p = np.inf
    for paper in alloc:
        util = sum([prices[rev] for rev in alloc[paper]])
        p = compute_p(paper, alloc, scores, alpha)
        if util < caps[paper] and p < i_p and paper not in disallowed_i:
            i = paper
            i_p = p
    return i


# Just represent the tight graph as a pair of dictionaries, one maps agents to their forward neighbors, and the
# other maps goods to their forward neighbors.
# def tight_graph(scores, alpha, prices, alloc):
#     n = len(alpha)
#     m = len(prices)
#
#     agent_neighbors = {a: set() for a in range(n)}
#     good_neighbors = {g: set() for g in range(m)}
#
#     for agent in range(n):
#         for good in range(m):
#             if good not in alloc[agent] and alpha[agent] == scores[good, agent]/prices[good]:
#                 agent_neighbors[agent].add(good)
#             if good in alloc[agent] and alpha[agent] == scores[good, agent]/prices[good]:
#                 good_neighbors[good].add(agent)
#
#     return agent_neighbors, good_neighbors


def tight_graph(scores, alpha, prices, alloc):
    n = len(alpha)
    m = len(prices)

    agents = [a for a in range(n)]
    goods = [g + n for g in range(m)]

    fwd_nbrs = {v: set() for v in range(n+m)}

    for agent in agents:
        for g in goods:
            good = g - n
            if good not in alloc[agent] and math.isclose(alpha[agent], scores[good, agent] / prices[good]):
                fwd_nbrs[agent].add(g)
            if good in alloc[agent] and math.isclose(alpha[agent], scores[good, agent]/prices[good]):
                fwd_nbrs[g].add(agent)

    return fwd_nbrs


def correct_offset(imp_path, n):
    # Correct for the offset of goods during graph construction
    imp = []
    for vertex in imp_path:
        if vertex >= n:
            vertex -= n
        imp.append(vertex)
    return imp


def bfs(i, tight_graph, alloc, epsilon, scores, alpha):
    m, n = scores.shape
    bfs_queue = [i]
    backptrs = {}
    i_p = compute_p(i, alloc, scores, alpha)
    visited_nodes = set()

    while len(bfs_queue) > 0:
        curr = bfs_queue.pop(0)

        visited_nodes.add(curr)
        nbrs = tight_graph[curr]
        for nbr in nbrs:
            if nbr not in visited_nodes:
                backptrs[nbr] = curr
                bfs_queue.append(nbr)

        # Check for the end condition and terminate if so
        if curr < n and curr != i:
            p_ah = compute_p(curr, alloc, scores, alpha) - scores[backptrs[curr]-n, curr] / alpha[curr]
            if p_ah > (1+epsilon) * i_p:
                # Found an improving path
                imp_path = []
                while curr in backptrs:
                    imp_path.insert(0, curr)
                    curr = backptrs[curr]
                imp_path.insert(0, i)
                return correct_offset(imp_path, n)

    return None

# Perform a BFS starting at i. Return the first "improving path" we find.
# def bfs(i, an, gn, alloc, epsilon, scores, alpha):
#     bfs_queue = [i]
#     search_radius_queue = [0]
#     agent_backptrs = {}
#     good_backptrs = {}
#     i_p = compute_p(i, alloc, scores, alpha)
#     visited_agents = {i}
#     visited_goods = set()
#
#     visited_sets = [visited_agents, visited_goods]
#     backptrs = [good_backptrs, agent_backptrs]
#     maps = [an, gn]
#
#     while len(bfs_queue) > 0:
#         curr = bfs_queue.pop(0)
#         search_radius = search_radius_queue.pop(0)
#
#         mod = search_radius % 2
#
#         visited_sets[mod].add(curr)
#         nbrs = maps[mod][curr]
#         for n in nbrs:
#             backptrs[mod][n] = curr
#             if n not in visited_sets[(mod + 1) % 2]:
#                 bfs_queue.append(n)
#                 search_radius_queue.append(search_radius + 1)
#
#         # if search_radius % 2 == 0:
#         #     # agent
#         #     visited_agents.add(curr)
#         #     nbrs = an[curr]
#         #     for n in nbrs:
#         #         good_backptrs[n] = curr
#         #         if n not in visited_goods:
#         #             bfs_queue.append(n)
#         #             search_radius_queue.append(search_radius + 1)
#         # else:
#         #     # good
#         #     visited_goods.add(curr)
#         #     nbrs = gn[curr]
#         #     for n in nbrs:
#         #         agent_backptrs[n] = curr
#         #         if n not in visited_agents:
#         #             bfs_queue.append(n)
#         #             search_radius_queue.append(search_radius + 1)
#
#         # Check for the end condition and terminate if so
#         if search_radius % 2 == 0 and search_radius > 0:
#             p_ah = compute_p(curr, alloc, scores, alpha) - scores[agent_backptrs[curr], curr]/alpha[curr]
#             if p_ah > (1+epsilon) * i_p:
#                 # Found an improving path
#                 imp_path = []
#                 while search_radius > 0 and \
#                         ((curr in agent_backptrs and search_radius % 2 == 0) or
#                          (curr in good_backptrs and search_radius % 2 == 1)):
#                     imp_path.insert(0, curr)
#                     if search_radius % 2 == 0:
#                         curr = agent_backptrs[curr]
#                     else:
#                         curr = good_backptrs[curr]
#                     search_radius -= 1
#                 imp_path.insert(0, i)
#                 return imp_path
#
#         search_radius += 1
#
#     return None


# Follow the improving path backward, reassigning goods as you go.
def perform_swaps(alloc, imp, epsilon, scores, alpha):
    comparison = (1+epsilon)*compute_p(imp[0], alloc, scores, alpha)
    while len(imp) > 1 and \
            compute_p(imp[-1], alloc, scores, alpha) - scores[imp[-2], imp[-1]]/alpha[imp[-1]] > comparison:

        alloc[imp[-1]].remove(imp[-2])
        alloc[imp[-3]].append(imp[-2])
        imp.pop(-1)
        imp.pop(-1)
    return alloc


def get_reachable(i, tg, n):
    queue = [i]
    visited_nodes = set()

    while len(queue):
        curr = queue.pop(0)

        visited_nodes.add(curr)
        for nbr in tg[curr]:
            if nbr not in visited_nodes:
                queue.append(nbr)

    visited_goods = {g - n for g in visited_nodes if g >= n}
    visited_agents = {a for a in visited_nodes if a < n}

    return visited_goods, visited_agents, set(range(len(tg)-n)) - visited_goods, set(range(n)) - visited_agents


# def get_reachable(i, tg):
#     queue = [i]
#     search_radius = 0
#     visited_goods = set()
#     visited_agents = {i}
#
#     while len(queue):
#         curr = queue.pop(0)
#         while len(queue) > 0:
#             curr = queue.pop(0)
#             if search_radius % 2 == 0:
#                 # agent
#                 visited_agents.add(curr)
#                 for n in an[curr]:
#                     if n not in visited_agents:
#                         queue.append(n)
#             else:
#                 # good
#                 visited_goods.add(curr)
#                 for n in gn[curr]:
#                     if n not in visited_goods:
#                         queue.append(n)
#
#         search_radius += 1
#
#     return visited_goods, visited_agents, set(range(len(gn))) - visited_goods, set(range(len(an))) - visited_agents


def compute_beta(reach_goods, reach_agents, unreach_goods, unreach_agents, prices, scores, alloc, alpha, r, i, caps):
    i_p = compute_p(i, alloc, scores, alpha)

    beta1 = np.inf
    for k in reach_agents:
        for j in unreach_goods:
            if j not in alloc[k] and scores[j, k] and prices[j]:
                beta1 = min(beta1, alpha[k]/(scores[j, k]/prices[j]))

    beta2 = np.inf
    for k in unreach_agents:
        for j in reach_goods:
            if j in alloc[k] and prices[j] and alpha[k]:
                beta2 = min(beta2, (scores[j, k]/prices[j])/alpha[k])

    if i_p > 0:
        beta3 = -np.inf
        for k in unreach_agents:
            pk = [scores[rev, k]/alpha[k] for rev in alloc[k]]
            if len(pk):
                beta3 = max(beta3, sum(pk) - max(pk))
        beta3 = (1/(r**2 * i_p)) * beta3

        h_p = np.inf
        for paper in unreach_agents:
            util = sum([scores[rev, paper] for rev in alloc[paper]])
            p = compute_p(paper, alloc, scores, alpha)
            if util < caps[paper] and p < h_p:
                h_p = p
        s_bound = np.log(h_p/i_p)/np.log(r)
        beta4 = r**(math.floor(s_bound)+1)

    else:
        beta3 = np.inf
        beta4 = np.inf

    beta = min(beta1, beta2, max(1, beta3), beta4)
    break_after = beta3 <= min(beta1, beta2, beta4)
    if beta < 1:
        if np.isclose(beta, 1):
            beta = 1
            # print("close beta1")
            # for k in reach_agents:
            #     for j in unreach_goods:
            #         if j not in alloc[k] and scores[j, k] and prices[j]:
            #             print(alpha[k], scores[j, k]/prices[j])
        # print()
        else:
            print(beta1, beta2, beta3, beta4)
            sys.exit(0)

        # print(i_p, alloc[i], )
    if beta == 1:
        print(beta1, beta2, beta3, beta4)
    return beta, break_after


def get_revert_labels_map(scores):
    m = scores.shape[0]
    label_map = {}

    num_valid_goods = 0
    for g in range(m):
        if np.max(scores[g, :]) > 0:
            label_map[num_valid_goods] = g
            num_valid_goods += 1
    return label_map


def revert_labels(alloc, prices, r_map):
    _alloc = {}
    for paper in alloc:
        _alloc[paper] = []
        for rev in alloc[paper]:
            _alloc[paper].append(r_map[rev])

    _prices = {}
    for rev in prices:
        _prices[r_map[rev]] = prices[rev]

    return _alloc, _prices


def get_chaudhury_alloc(loads, scores, caps, epsilon):
    # Remove goods with 0 value, save a map that will let us relabel
    # the goods back to their original labels when we finish
    revert_labels_map = get_revert_labels_map(scores)
    scores = scores[np.where(np.max(scores, axis=1) > 0)]
    loads = loads[np.where(np.max(scores, axis=1) > 0)]

    # Convert scores to be 0 if negative (no "chores")
    scores = np.maximum(scores, np.zeros(scores.shape))

    # Round the valuations to be powers of 1 + epsilon
    r = 1 + epsilon
    scores, caps = perform_rounding(scores, caps, r)

    # Cap the valuations (just keep track of dups during the algorithm e.g. by setting values to 0 if dups)
    scores = np.minimum(caps.reshape(1, -1) * np.ones(scores.shape), scores)

    # Greedy initial assignment, price setting
    alloc, prices = greedy_initial_assignment(loads, scores)

    # Set MBB ratios
    alpha = np.ones(caps.shape)

    disallowed_i = set()

    loop = 0
    # old_alloc = deepcopy(alloc)
    # old_alpha = deepcopy(alpha)
    while True:
        # check epsilon-p-EF1, maybe finish
        if eps_p_ef1(alloc, epsilon, scores=scores, alpha=alpha):
            print("nsw at end1: ", nsw(alloc,scores))
            return revert_labels(alloc, prices, revert_labels_map)

        # Select i
        i = select_i(alloc, prices, caps, scores, alpha, disallowed_i)
        print("i: ", i)
        loop +=1
        if loop % 10 == 0:
            nw = nsw(alloc, scores)
            print("nsw: ", nw)
            if nw > 2.02:
                return revert_labels(alloc, prices, revert_labels_map)


        # BFS for an improving path
        tg = tight_graph(scores, alpha, prices, alloc)
        shortest_improving_path = bfs(i, tg, alloc, epsilon, scores, alpha)
        # print(shortest_improving_path)

        if shortest_improving_path:
            alloc = perform_swaps(alloc, shortest_improving_path, epsilon, scores, alpha)
            # disallowed_i = set()
        else:
            # Update prices and MBB ratios
            reach_goods, reach_agents, unreach_goods, unreach_agents = get_reachable(i, tg, len(caps))
            beta, break_after = compute_beta(reach_goods, reach_agents, unreach_goods, unreach_agents, prices, scores, alloc, alpha, r, i, caps)
            print(beta)

            # This was a hack to avoid infinite loops. Probably shouldn't occur ever.
            # if beta == 1:
            #     disallowed_i.add(i)

            # Update prices and MBB ratios
            for g in reach_goods:
                prices[g] *= beta

            for a in reach_agents:
                alpha[a] /= beta

            if break_after:
                print("nsw at end: ", nsw(alloc, scores))
                return revert_labels(alloc, prices, revert_labels_map)

        # loop += 1
        # if loop % 1 == 0:
        #     print(loop)
        #     if old_alloc == alloc and np.all(alpha == old_alpha):
        #         print("problem")
        #         print(i)
        #         print(shortest_improving_path)
        #         print(beta)
        #     elif old_alloc == alloc:
        #         print(old_alpha)
        #         print(alpha)
        #     old_alloc = deepcopy(alloc)
        #     old_alpha = deepcopy(alpha)


def drop_revs(alloc, scores, covs):
    _alloc = {}
    for paper, rev_set in alloc.items():
        top_revs = sorted(rev_set, key=lambda r: scores[r, paper], reverse=True)
        _alloc[paper] = top_revs[:min(len(rev_set), covs[paper])]
    return _alloc


def print_stats(alloc, paper_reviewer_affinities, covs):
    print("usw: ", usw(alloc, paper_reviewer_affinities))
    print("nsw: ", nsw(alloc, paper_reviewer_affinities))
    print("ef1 violations: ", ef1_violations(alloc, paper_reviewer_affinities))
    print("efx violations: ", efx_violations(alloc, paper_reviewer_affinities))
    print("paper coverage violations: ", paper_coverage_violations(alloc, covs))
    print("reviewer load distribution: ", reviewer_load_distrib(alloc))
    print("paper scores: ", paper_score_stats(alloc, paper_reviewer_affinities))
    print()


if __name__ == "__main__":
    # We want to obtain a 1.44-NSW allocation, which should also be 4-epsilon price envy free up to 1 item.

    dataset = sys.argv[1]
    timestamps = {"cvpr": "2020-09-16-10-28-42", "cvpr2018": "2020-09-16-10-26-09", "midl": "2020-09-16-09-32-53"}

    paper_reviewer_affinities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/scores.npy" % dataset)
    reviewer_loads = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/loads.npy" % dataset)
    paper_capacities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/covs.npy" % dataset)

    start = time.time()
    epsilon = 1/5
    alloc, prices = get_chaudhury_alloc(reviewer_loads, paper_reviewer_affinities, paper_capacities, epsilon)
    print("Correctly satisfies the approximate EF1 notion: ", eps_p_ef1(alloc, epsilon*4, prices=prices))
    runtime = time.time() - start

    print(alloc)

    print("chaudhury Algorithm Results")
    print("%.2f seconds" % runtime)
    print_stats(alloc, paper_reviewer_affinities, paper_capacities)

    # Try dropping reviewers from papers to meet constraints
    alloc = drop_revs(alloc, paper_reviewer_affinities, paper_capacities)

    print(alloc)

    print("chaudhury (Meeting Constraints) Results")
    print("%.2f seconds" % runtime)
    print_stats(alloc, paper_reviewer_affinities, paper_capacities)

    # Load the fairflow solution for MIDL without the reviewer lower bounds...
    fairflow_soln = load_fairflow_soln("/home/justinspayan/Fall_2020/fair-matching/exp_out/%s/fairflow/"
                                       "%s/results/assignment.npy" % (dataset, timestamps[dataset]))
    print("Fairflow Results")
    print_stats(fairflow_soln, paper_reviewer_affinities, paper_capacities)
    print("***********\n***********\n")

