import time

from gurobipy import *

from utils import *


def create_multidict(pra):
    d = {}
    for rev in range(pra.shape[0]):
        for paper in range(pra.shape[1]):
            d[(paper, rev)] = pra[rev, paper]
    return multidict(d)


def add_vars_to_model(m, paper_rev_pairs):
    x = m.addVars(paper_rev_pairs, name="assign", vtype=GRB.BINARY, lb=0.0, ub=1.0)  # The binary assignment variables
    return x


def add_constrs_to_model(m, x, covs, loads, matrix_alloc):
    papers = range(covs.shape[0])
    revs = range(loads.shape[0])
    m.addConstrs((x.sum(paper, '*') == (covs[paper] - np.sum(matrix_alloc[:, paper])) for paper in papers),
                 'covs')  # Paper coverage constraints
    m.addConstrs((x.sum('*', rev) <= (loads[rev] - np.sum(matrix_alloc[rev, :])) for rev in revs),
                 'loads')  # Reviewer load constraints
    m.addConstrs((x[p, r] <= 1 - matrix_alloc[r, p] for r, p in product(revs, papers)))


def construct_equivalence_classes(scores):
    ec = {}
    for agent in range(scores.shape[1]):
        agent_ec = {}
        current_value = np.inf
        agent_rev_scores = np.sort(scores[:, agent])[::-1].tolist()
        revs_in_order = np.argsort(scores[:, agent])[::-1].tolist()

        for score, rev in zip(agent_rev_scores, revs_in_order):
            if score < current_value:
                current_value = score
                agent_ec[len(agent_ec)] = []
            agent_ec[len(agent_ec) - 1].append(rev)
        ec[agent] = agent_ec
    return ec


"""Maximize the usw if we start with this allocation, add r to a, then just find the best USW respecting the remaining
constraints. Use network flow to do this."""


def max_usw_possible(new_r, new_a, matrix_alloc, loads, covs, scores):
    paper_rev_pairs, pras = create_multidict(scores)

    m = Model("max_usw")

    x = add_vars_to_model(m, paper_rev_pairs)
    new_pair = np.zeros(matrix_alloc.shape)
    new_pair[new_r, new_a] = 1
    add_constrs_to_model(m, x, covs, loads, matrix_alloc + new_pair)

    m.setObjective(x.prod(pras), GRB.MAXIMIZE)
    m.optimize()

    return m.objVal + np.sum(scores * (matrix_alloc + new_pair))


def simple_greedy_usw(r, a, matrix_alloc, loads, covs, scores):
    ma_copy = matrix_alloc.copy()
    ma_copy[r, a] += 1

    num_allocs = np.sum(matrix_alloc)
    target = np.sum(covs)

    while num_allocs < target:
        allowed = 1 - ma_copy
        allowed *= (loads.reshape(-1, 1) - np.sum(ma_copy, axis=1).reshape(-1, 1))
        allowed *= (covs.reshape(1, -1) - np.sum(ma_copy, axis=0))
        allowed[np.nonzero(allowed)] = 1
        selection = np.argmax(scores * allowed)
        ma_copy[np.unravel_index(selection, scores.shape)] += 1
        num_allocs += 1

    return np.sum(ma_copy * scores)


"""Check if we can assign this reviewer to this paper under the constraints. If so, then you need 
to check if we can attain a USW of w in the final allocation if we do so. First greedily, then using an LP or
flow or something."""


def can_assign(r, a, matrix_alloc, loads, covs, scores, w):
    # Should never happen
    assert (np.sum(matrix_alloc[:, a]) < covs[a])

    if np.sum(matrix_alloc[r, :]) >= loads[r] or matrix_alloc[r, a]:
        return False

    best_usw = max_usw_possible(r, a, matrix_alloc, loads, covs, scores) / matrix_alloc.shape[1]
    return best_usw >= w


"""Constrained Round-Robin from https://arxiv.org/pdf/1908.00161.pdf."""


def crr(scores, loads, covs, best_revs, alloc_file, w):
    m, n = scores.shape

    poorest_agents = set(range(n))
    matrix_alloc = np.zeros((scores.shape))

    ec = construct_equivalence_classes(scores)

    while np.sum(matrix_alloc) < np.sum(covs):
        success = False
        for a in sorted(poorest_agents):
            success = False
            # Try to take from this agent's best equivalence class. If we succeed, break and move on.
            # If not, move to next agent
            if ec[a]:
                best_ec = sorted(ec[a].keys())[0]
                for r in ec[a][best_ec]:
                    if can_assign(r, a, matrix_alloc, loads, covs, scores, w):
                        matrix_alloc[r, a] += 1
                        ec[a][best_ec].remove(r)
                        if not ec[a][best_ec]:
                            del ec[a][best_ec]
                        if not ec[a]:
                            del ec[a]
                        success = True
                        break

            if success:
                break

        if not success:
            for a in sorted(poorest_agents):
                if ec[a]:
                    best_ec = sorted(ec[a].keys())[0]
                    del ec[a][best_ec]
                    if not ec[a]:
                        del ec[a]

        # if we went through all poorest_agents and haven't assigned a good, remove their top equivalence
        #  class and go back

        min_revs_assigned = np.min(np.sum(matrix_alloc, axis=0))
        poorest_agents = [a for a in range(n) if np.sum(matrix_alloc[:, a]) == min_revs_assigned and ec[a]]
        # update the "poorest_agents"

    alloc = {p: set() for p in range(n)}
    for p in range(n):
        for r in range(m):
            if matrix_alloc[r, p]:
                alloc[p].add(r)
    return alloc


def run_algo(dataset, base_dir, alloc_file, w):
    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)

    best_revs = np.argsort(-1 * scores, axis=0)

    alloc = crr(scores, loads, covs, best_revs, alloc_file, w)
    return alloc


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = args.base_dir
    alloc_file = args.alloc_file
    w = args.w_value

    random.seed(args.seed)

    start = time.time()
    alloc = run_algo(dataset, base_dir, alloc_file, w)
    runtime = time.time() - start

    save_alloc(alloc, alloc_file)

    print("Final CRR Results")
    print("%.2f seconds" % runtime)

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy"))

    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, paper_capacities)
