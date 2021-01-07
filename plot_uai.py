import json
import jsonlines
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.approximation import treewidth
import os

from collections import defaultdict
from itertools import combinations


def analyze_graph(G, name):
    tw, _ = treewidth.treewidth_min_fill_in(G)
    nx.draw_random(G)
    plt.savefig(name)
    print("Treewidth: %s" % tw)
    print("Node degrees")
    print(sorted(G.degree, key=lambda x: x[1]))


def plot_non_hierarchically(submission_dir):
    nodes = set()
    edges = set()

    for jsonfile in os.listdir(submission_dir):
        with open(os.path.join(submission_dir, jsonfile)) as f:
            paper = json.load(f)
            sas = paper["content"]["subject areas"]
            nodes = nodes.union(set(sas))
            for p in combinations(sas, 2):
                edges.add(p)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    analyze_graph(G, "uai_18_non_hier.png")


def plot_must_share_fraction(submission_dir, overlap_threshold):
    nodes = set()
    edges = set()

    groups_to_papers = defaultdict(set)
    i = 0
    for jsonfile in os.listdir(submission_dir):
        with open(os.path.join(submission_dir, jsonfile)) as f:
            paper = json.load(f)
            sas = paper["content"]["subject areas"]
            nodes = nodes.union(set(sas))
            # for p in combinations(sas, 2):
            #     edges.add(p)
            for sa in sas:
                groups_to_papers[sa].add(i)
        i += 1

    for sa in groups_to_papers:
        for sa2 in groups_to_papers:
            if sa < sa2:
                olap = groups_to_papers[sa] & groups_to_papers[sa2]
                if len(olap)/len(groups_to_papers[sa]) > overlap_threshold and \
                        len(olap)/len(groups_to_papers[sa2]) > overlap_threshold:
                    edges.add((sa, sa2))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    analyze_graph(G, "uai_18_groups_thresh.png")


def plot_papers(submission_dir, overlap_threshold):
    nodes = set()
    edges = set()

    i = 0
    paper_subject_areas = {}
    for jsonfile in os.listdir(submission_dir):
        with open(os.path.join(submission_dir, jsonfile)) as f:
            paper = json.load(f)
            sas = paper["content"]["subject areas"]
            nodes.add(i)
            i = i + 1
            paper_subject_areas[i] = set(sas)
    for paper in paper_subject_areas:
        for paper2 in paper_subject_areas:
            if paper < paper2:
                olap = paper_subject_areas[paper] & paper_subject_areas[paper2]
                if len(olap)/len(paper_subject_areas[paper]) > overlap_threshold and \
                        len(olap)/len(paper_subject_areas[paper2]) > overlap_threshold:
                    edges.add((paper, paper2))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # print(G.edges())

    analyze_graph(G, "uai_18_papers.png")


def plot_pairwise(submission_dir):
    nodes = set()
    edges = set()

    for jsonfile in os.listdir(submission_dir):
        with open(os.path.join(submission_dir, jsonfile)) as f:
            paper = json.load(f)
            sas = paper["content"]["subject areas"]
            # print(sas)
            # print(list(combinations(set(combinations(sas, 3)), 2)))
            nodes = nodes.union(set(combinations(sas, 2)))
            for p in combinations(set(combinations(sas, 2)), 2):
                if len(set(p[0]) & set(p[1])) == 0:
                    # print(p)
                    edges.add(p)

    print(len(nodes))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    analyze_graph(G, "uai_18_pairwise.png")


def plot_hierarchically(submission_dir):
    pass


def plot_reviewer_bids(bid_file):
    nodes = set()
    edges = set()

    reviewers_to_preferred_papers = defaultdict(set)
    with jsonlines.open(bid_file) as reader:
        for obj in reader:
            paper_id = obj["forum"]
            bid = obj["tag"]
            reviewer_id = obj["readers"][0]

            nodes.add(reviewer_id)

            if bid == "I can review":
                reviewers_to_preferred_papers[reviewer_id].add(paper_id)

    for rid1, rid2 in combinations(reviewers_to_preferred_papers, 2):
        olap = reviewers_to_preferred_papers[rid1] & reviewers_to_preferred_papers[rid2]
        if olap:
            print(olap)
            print(rid1, rid2)
            edges.add((rid1, rid2))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    analyze_graph(G, "uai_18_reviewer_bids.png")


if __name__ == "__main__":
    submission_dir = "/home/justinspayan/ExpertiseModeling/uai18/submission_records_fulltext"
    # plot_non_hierarchically(submission_dir)
    # plot_pairwise(submission_dir)
    # plot_papers(submission_dir, .9)
    plot_must_share_fraction(submission_dir, .15)
    # bid_file = "/home/justinspayan/ExpertiseModeling/uai18/reviewer_bids.jsonl"
    # plot_reviewer_bids(bid_file)