import math
import networkx as nx
import pickle

from usw_and_ef1 import *


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


def select_i(alloc, prices, caps, scores, alpha):
    i = None
    i_p = np.inf
    for paper in alloc:
        # We want the least-spending, uncapped agent.
        bundle_sz = len(alloc[paper])
        p = compute_p(paper, alloc, scores, alpha)
        if bundle_sz < caps[paper] and p < i_p:
            i = paper
            i_p = p
    return i


def tight_graph(scores, alpha, prices, alloc):
    n = len(alpha)
    m = len(prices)

    prices = np.array([prices[g] for g in range(len(prices))])

    agents = [a for a in range(n)]
    goods = [g + n for g in range(m)]

    fwd_nbrs = {v: set() for v in range(n+m)}

    for agent in agents:
        edges_to_goods = np.where(np.isclose(alpha[agent], scores[:, agent]/prices))[0]
        if edges_to_goods.shape[0] > 0:
            for good in np.nditer(edges_to_goods):
                if good not in alloc[agent]:
                    fwd_nbrs[agent].add(good+n)

        for g in alloc[agent]:
            if math.isclose(alpha[agent], scores[g, agent]/prices[g]):
                fwd_nbrs[g+n].add(agent)

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
                return correct_offset(imp_path, n), visited_nodes, bfs_queue

    return None, visited_nodes, bfs_queue


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


def get_reachable(tg, n, visited_nodes, queue):
    while len(queue):
        curr = queue.pop(0)

        visited_nodes.add(curr)
        for nbr in tg[curr]:
            if nbr not in visited_nodes:
                queue.append(nbr)

    visited_goods = {g - n for g in visited_nodes if g >= n}
    visited_agents = {a for a in visited_nodes if a < n}

    return visited_goods, visited_agents, set(range(len(tg)-n)) - visited_goods, set(range(n)) - visited_agents


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
        h = None
        for paper in unreach_agents:
            p = compute_p(paper, alloc, scores, alpha)
            if len(alloc[paper]) < caps[paper] and p < h_p:
                h_p = p
                h = paper
        if h is not None:
            s_bound = np.log(h_p/i_p)/np.log(r)
            beta4 = r**(math.floor(s_bound)+1)
        else:
            beta4 = np.inf

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


def nsw(alloc, pra, caps=None):
    if caps is not None:
        nsw = 1
        for p in alloc:
            paper_scores = [pra[r, p] for r in alloc[p]]
            paper_scores = sorted(paper_scores, reverse=True)[:caps[p]]
            nsw *= sum(paper_scores)**(1/len(alloc))
        return nsw
    else:
        nsw = 1
        for p in alloc:
            paper_score = 0
            for r in alloc[p]:
                paper_score += pra[r, p]
            nsw *= paper_score ** (1 / len(alloc))
        return nsw


def get_barman_alloc(loads, scores, caps, epsilon):
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
    # scores = np.minimum(caps.reshape(1, -1) * np.ones(scores.shape), scores)

    # Greedy initial assignment, price setting
    alloc, prices = greedy_initial_assignment(loads, scores)

    # Set MBB ratios
    alpha = np.ones(caps.shape)

    loop = 0
    # old_alloc = deepcopy(alloc)
    # old_alpha = deepcopy(alpha)
    while True:
        # check epsilon-p-EF1, maybe finish
        if eps_p_ef1(alloc, epsilon, scores=scores, alpha=alpha):
            print("nsw at end1: ", nsw(alloc, scores, caps))
            return revert_labels(alloc, prices, revert_labels_map)

        # Select i
        i = select_i(alloc, prices, caps, scores, alpha)
        print("i: ", i)
        if i is None:
            # Another stopping condition. If all agents are capped, that means we cannot improve.
            return revert_labels(alloc, prices, revert_labels_map)
        loop +=1
        if loop % 10 == 0:
            nw = nsw(alloc, scores, caps)
            print("capped nsw: ", nw)
            # Early stopping, since my impl
            # apparently hits an infinite loop on CVPR somewhere otherwise :(
            # On second thought, it may not even be a loop but rather a really intense search in the graph.
            if nw > 25:
                return revert_labels(alloc, prices, revert_labels_map)

        # BFS for an improving path
        tg = tight_graph(scores, alpha, prices, alloc)
        shortest_improving_path, bfs_visited_nodes, bfs_queue = bfs(i, tg, alloc, epsilon, scores, alpha)
        # print(shortest_improving_path)

        if shortest_improving_path:
            alloc = perform_swaps(alloc, shortest_improving_path, epsilon, scores, alpha)
            # disallowed_i = set()
        else:
            # Update prices and MBB ratios
            reach_goods, reach_agents, unreach_goods, unreach_agents = get_reachable(tg, len(caps), bfs_visited_nodes, bfs_queue)
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
                print("nsw at end: ", nsw(alloc, scores, caps))
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


def add_revs(alloc, paper_reviewer_affinities, paper_capacities, reviewer_load_caps):
    _alloc = {}
    rev_loads = Counter()
    for rev_set in alloc.values():
        for r in rev_set:
            rev_loads[r] += 1

    for paper, rev_set in alloc.items():
        best_revs = np.argsort(paper_reviewer_affinities[:, paper])[::-1].tolist()
        while len(rev_set) < paper_capacities[paper]:
            r = best_revs.pop(0)
            if r not in rev_set and rev_loads[r] < reviewer_load_caps[r]:
                rev_set.append(r)
                rev_loads[r] += 1
        _alloc[paper] = rev_set

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


def run_algo(dataset, epsilon):
    paper_reviewer_affinities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/scores.npy" % dataset)
    reviewer_loads = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/loads.npy" % dataset).astype(np.int64)
    paper_capacities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/covs.npy" % dataset).astype(np.int64)

    alloc, prices = get_barman_alloc(reviewer_loads, paper_reviewer_affinities, paper_capacities, epsilon)
    print("Correctly satisfies the approximate EF1 notion: ", eps_p_ef1(alloc, epsilon * 4, prices=prices))
    return alloc, prices


def save_alloc(alloc, prices, dataset, msg):
    with open("%s_item_limits_alloc_%s" % (dataset, msg), 'wb') as f:
        pickle.dump(alloc, f)
    with open("%s_item_limits_prices_%s" % (dataset, msg), 'wb') as f:
        pickle.dump(prices, f)


if __name__ == "__main__":
    # We want to obtain a 1.44-NSW allocation, which should also be 4-epsilon price envy free up to 1 item.
    dataset = sys.argv[1]
    dry_run = sys.argv[2]
    if dry_run != "dry":
        epsilon = 1/5
        start = time.time()
        alloc, prices = run_algo(dataset, epsilon)
        runtime = time.time() - start

        # with open('cvpr_item_limits_alloc', 'r') as inf:
        #     alloc = eval(inf.read())

        print(alloc)

        save_alloc(alloc, prices, dataset, "")

        print("Barman Algorithm Results")
        print("%.2f seconds" % runtime)
        print_stats(alloc, paper_reviewer_affinities, paper_capacities)

        # Drop and then add (?) reviewers from papers to meet constraints
        alloc = drop_revs(alloc, paper_reviewer_affinities, paper_capacities)
        alloc = add_revs(alloc, paper_reviewer_affinities, paper_capacities, reviewer_loads)

        print(alloc)

        save_alloc(alloc, prices, dataset, "cov_corr")

        print("Barman-Item-Caps (Meeting Constraints) Results")
        print("%.2f seconds" % runtime)
        print_stats(alloc, paper_reviewer_affinities, paper_capacities)

    paper_reviewer_affinities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/scores.npy" % dataset)
    paper_capacities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/covs.npy" % dataset)

    # Load the fairflow solution without the reviewer lower bounds...
    timestamps = {"cvpr": "2020-09-16-10-28-42", "cvpr2018": "2020-09-16-10-26-09", "midl": "2020-09-16-09-32-53"}

    fairflow_soln = load_fairflow_soln("/home/justinspayan/Fall_2020/fair-matching/exp_out/%s/fairflow/"
                                       "%s/results/assignment.npy" % (dataset, timestamps[dataset]))
    print("Fairflow Results")
    print_stats(fairflow_soln, paper_reviewer_affinities, paper_capacities)
    print("***********\n***********\n")

    # FairIR solutions
    timestamps = {"cvpr": "2021-01-07-12-07-19", "cvpr2018": "2021-01-07-12-29-03", "midl": "2020-09-18-14-42-49"}

    fairir_soln = load_fairflow_soln("/home/justinspayan/Fall_2020/fair-matching/exp_out/%s/fairir/"
                                       "%s/results/assignment.npy" % (dataset, timestamps[dataset]))
    print("FairIR Results")
    print_stats(fairir_soln, paper_reviewer_affinities, paper_capacities)
    print("***********\n***********\n")

