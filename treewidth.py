import networkx as nx
from networkx.algorithms.approximation import treewidth

from eit_by_agents import ef1b_prop_nonoverlapping, brute_force_solve


def treewidth_vs_points_dropped():
    for n in range(4, 10):
        groups = {}
        idx = 0
        for i in range(n):
            # Add the K_n in the center
            groups[idx] = set(range(n)) - {i}
            # Add the additional element attaching to all but one element of K_n
            idx += 1
            groups[idx] = {n, i}
            idx += 1

        edges = []
        for idx, group in groups.items():
            for idx2, group2 in groups.items():
                if idx < idx2 and (group & group2):
                    edges.append((idx, idx2))

        G = nx.Graph()
        G.add_nodes_from(groups)
        # print(G.nodes())
        G.add_edges_from(edges)

        # print(G.edges())
        tw, _ = treewidth.treewidth_min_fill_in(G)
        # print(tw)

        max_dist = 0
        dist = 0
        for points in range(10,100):
            problem_instance = (n+1, points, [list(g) for g in groups.values()], ef1b_prop_nonoverlapping)
            failed = sum(brute_force_solve(*problem_instance)) != points
            print(points, not failed)
            if failed:
                dist += 1
                max_dist = max([dist, max_dist])
            else:
                dist = 0
        print(n, tw, max_dist)


def treewidth_of_lower_bound_exs():
    num_copies = 1
    n = 4
    groups = {}
    idx = 0
    for c in range(num_copies):
        for i in range(n):
            # Add the K_n in the center
            groups[idx] = set(range((n+1)*c, (n+1)*c + n)) - {(n+1)*c + i}
            # Add the additional element attaching to all but one element of K_n
            idx += 1
            groups[idx] = {(n + 1) * c + n, (n + 1) * c + i}
            # print(c, i, idx)
            # print(groups[idx])
            if c > 0 and i == 0:
                groups[idx].add((n + 1) * (c-1) + n)
            elif c < num_copies -1 and i == n-1:
                groups[idx].add((n + 1) * (c+1) + n)
            idx += 1

    edges = []
    for idx, group in groups.items():
        for idx2, group2 in groups.items():
            if idx < idx2 and (group & group2):
                edges.append((idx, idx2))

    G = nx.Graph()
    G.add_nodes_from(groups)
    print(groups)
    print(G.nodes())
    G.add_edges_from(edges)
    print(G.edges())
    # nx.draw(G)

    tw, decomp = treewidth.treewidth_min_fill_in(G)
    print(tw)
    print(decomp.nodes())
    print(decomp.edges())

    points = 7
    problem_instance = ((n + 1) * num_copies, points, [list(g) for g in groups.values()], ef1b_prop_nonoverlapping)
    print(problem_instance)
    failed = sum(brute_force_solve(*problem_instance)) != points
    print(points, failed)

    from eit_by_agents import has_envious_pair
    print(has_envious_pair([list(g) for g in groups.values()], [2,1,1,1,2], ef1b_prop_nonoverlapping))

    # max_dist = 0
    # dist = 0
    # for points in range(5,20):
    #     problem_instance = ((n+1)*num_copies, points, [list(g) for g in groups.values()], ef1b_prop_nonoverlapping)
    #     print(problem_instance)
    #     failed = sum(brute_force_solve(*problem_instance)) != points
    #     print(points, not failed)
    #     if failed:
    #         dist += 1
    #         max_dist = max([dist, max_dist])
    #     else:
    #         dist = 0
    # print(n, tw, max_dist)


if __name__ == "__main__":
    treewidth_of_lower_bound_exs()