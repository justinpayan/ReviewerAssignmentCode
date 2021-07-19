import networkx as nx

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


def get_usw_alloc(reviewer_loads, paper_reviewer_affinities, paper_capacities):
    # 1d numpy array, 2d array, 1d numpy array
    g = construct_graph(reviewer_loads, paper_reviewer_affinities, paper_capacities)

    flow_dict = nx.min_cost_flow(g)

    # f = maximum_flow(g, 0, 1)
    # flow_value = f.flow_value
    # residuals = f.residual

    alloc = get_alloc_from_flow_result(flow_dict, reviewer_loads, paper_capacities)

    return alloc


def print_stats(alloc, paper_reviewer_affinities, covs):
    print("usw: ", usw(alloc, paper_reviewer_affinities))
    print("ef1 violations: ", ef1_violations(alloc, paper_reviewer_affinities))
    print("efx violations: ", efx_violations(alloc, paper_reviewer_affinities))
    print("paper coverage violations: ", paper_coverage_violations(alloc, covs))
    print("reviewer load distribution: ", reviewer_load_distrib(alloc))
    print("paper scores: ", paper_score_stats(alloc, paper_reviewer_affinities))
    print()


if __name__ == "__main__":
    # In this script, we will run the min-cost flow problem described in Ari's paper.
    # This will be the maximum USW solution for the AMG-SUB (addv marg gain = AMG) valuations.
    # We can see if this is EF1 or EFX, how many paper constraints it violates, etc.

    dataset = sys.argv[1]
    timestamps = {"cvpr": "2020-09-16-10-28-42", "cvpr2018": "2020-09-16-10-26-09", "midl": "2020-09-16-09-32-53"}

    paper_reviewer_affinities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/scores.npy" % dataset)
    reviewer_loads = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/loads.npy" % dataset)
    paper_capacities = np.load("/home/justinspayan/Fall_2020/fair-matching/data/%s/covs.npy" % dataset)

    start = time.time()
    alloc = get_usw_alloc(reviewer_loads, paper_reviewer_affinities, paper_capacities)
    runtime = time.time() - start

    print(alloc)

    print("Min-cost-flow-based Algorithm Results")
    print("%.2f seconds" % runtime)
    print_stats(alloc, paper_reviewer_affinities, paper_capacities)

    # Load the fairflow solution for MIDL without the reviewer lower bounds...
    fairflow_soln = load_fairflow_soln("/home/justinspayan/Fall_2020/fair-matching/exp_out/%s/fairflow/"
                                       "%s/results/assignment.npy" % (dataset, timestamps[dataset]))
    print("Fairflow Results")
    print_stats(fairflow_soln, paper_reviewer_affinities, paper_capacities)
    print("***********\n***********\n")

