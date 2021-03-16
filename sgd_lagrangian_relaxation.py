import os
import random
import time
import torch

from torch import nn
from utils import *


class SGDSearcher(object):
    def __init__(self, scores, loads, covs):
        self.scores = torch.tensor(scores).type(torch.FloatTensor)
        # Duplicate the reviewers "loads" times
        self.scores = self.scores.tile((loads, 1))
        self.loads = loads
        self.covs = covs
        self.m, self.n = scores.shape
        self.lr = 5e-1
        self.diag_mask = torch.ones((self.n, self.n)) - torch.eye(self.n)
        # self.mult_se = 1
        # self.mult_cov = 1
        # self.mult_rev = 1

    def convert_to_alloc(self, matrix_alloc):
        alloc = {i: list() for i in range(self.n)}
        agents_getting_each_item = np.argmax(matrix_alloc, axis=1)
        for g in range(self.m * self.loads):
            if np.sum(matrix_alloc[g, :]) > 1e-9:  # This needs to be checked, since we may have 0'd out that item.
                alloc[agents_getting_each_item[g]].append(g % self.m)
        return alloc

    # take the top self.n * self.covs elements of assn_priorities. Assn_base parametrizes a randomized allocation
    # with 1 good randomly assigned to some agent per row. You zero out the assignment for rows which are not
    # top n*c for assn_priorities.
    def assn_base_to_feasible_assn(self, assn_base, assn_priorities):
        assignment = nn.functional.softmax(assn_base, dim=1)
        # assn_priorities = nn.functional.sigmoid(assn_priorities)

        mask = torch.zeros(self.m * self.loads)
        mask[assn_priorities.topk(self.n * self.covs).indices] = 1
        return assignment * mask.reshape((-1, 1))

    def compute_strong_envy(self, allocation):
        # print(allocation.type())
        # print(self.scores.type())
        # The ij element should be i's valuation for j's bundle
        W = torch.matmul(torch.transpose(self.scores, 0, 1), allocation)
        # I want n matrices, where each matrix i has rows j which hold the value of i for items given to j.
        # M = (self.scores.reshape((1, self.m, self.n)) * allocation.reshape((self.m, 1, self.n))).transpose(0, 1)
        # M = torch.max(M, dim=2).values
        M = self.scores.transpose(0, 1).reshape(self.n, self.m * self.loads, 1) * allocation.reshape(1, self.m * self.loads, self.n)
        M = torch.max(M, dim=1).values
        # M's ij element is most valuable item for i in j's bundle. If we just change it so that all items not owned
        # by j are worth 1000 and then take the min, this will be EFX instead of EF1.
        strong_envy = (W - M) - torch.diag(W).reshape((-1, 1))
        strong_envy = torch.clamp(strong_envy, min=0)
        strong_envy = strong_envy * self.diag_mask
        return strong_envy

    def compute_rev_dup_violations(self, allocation):
        # Compute a matrix which, for each reviewer-paper pair calculates the total mass for assigning
        # that reviewer to that paper. We will require all elements of this matrix to be <= 1
        # print("allocation: ", allocation.shape)
        stack_alloc = allocation.reshape((self.loads, self.m, self.n))
        # print("stack_alloc: ", stack_alloc.shape)
        rev_load_per_agent = stack_alloc.sum(dim=0)
        # print("rev_load_per_agent: ", rev_load_per_agent.shape)
        return torch.clamp(rev_load_per_agent - 1, min=0)
        # # Determine the number of duplications of each reviewer. We want this number to be 1
        # which_agent = torch.argmax(allocation, dim=1) # Determine which agent we'd assign to
        # which_agent[torch.sum(allocation, dim=1) == 0] = -1 # We aren't assigning these items to any agent.
        # agent_assnts = which_agent.reshape((-1, self.m))
        # # Count unique, ignoring -1.

    def compute_paper_cov_violations(self, allocation):
        paper_coverage = torch.sum(allocation, dim=0)
        # return torch.abs(self.covs - paper_coverage)
        return self.covs - paper_coverage

    def compute_objective(self, assn_base, assn_priorities, u_k, u_r, u_p):
        # Convert x_k into a randomized allocation using just softmax
        # Or maybe we should convert x_k into a deterministic allocation using gumbel softmax... let's see.
        # If we do gumbel softmax, I think that would mean we are maximizing the expected value of the objective.
        # That could work... but first just see if using softmax alone/softmax plus a penalty on entropy will give
        # a deterministic allocation in the end.
        # allocation = nn.functional.softmax(x_k, dim=1)
        # allocation = nn.functional.gumbel_softmax(x_k, dim=1, hard=True)
        allocation = self.assn_base_to_feasible_assn(assn_base, assn_priorities)
        usw = torch.sum(allocation * self.scores)

        # Compute the penalty based on strong envy and the lagrange multiplier for each envy comparison
        strong_envy = self.compute_strong_envy(allocation)
        se_penalty = torch.sum(u_k * strong_envy)

        # Add penalty for paper coverage violations
        paper_cov_violations = self.compute_paper_cov_violations(allocation)
        cov_penalty = torch.sum(u_p * paper_cov_violations)

        # Add penalty for reviewer duplication violations
        rev_dup_violations = self.compute_rev_dup_violations(allocation)
        rev_penalty = torch.sum(u_r * rev_dup_violations)

        return usw - se_penalty - cov_penalty - rev_penalty, usw.data, se_penalty.data, cov_penalty.data, rev_penalty.data

    def run_lagrangian_relaxation(self):
        lam_k = 2
        u_k = torch.ones((self.n, self.n)) * self.diag_mask
        u_r = torch.ones((self.m, self.n))  # lagrange multipliers to force mass of a reviewer-paper pair to be <= 1
                                            # (if the solution is integral, this will imply no reviewer is assigned
                                            # twice to the same paper)
        u_p = torch.ones(self.n)  # lagrange multipliers to force each paper to get c reviewers

        allocation = None

        outer_tol = 5
        outer_prev_obj = np.inf
        counter = 0

        for outer_iter in range(100):
            # Find the x_k that maximizes the penalized USW
            assignment_base = torch.autograd.Variable(torch.rand(self.scores.shape), requires_grad=True)
            # Determines which good assignment vectors to actually use
            assignment_priorities = torch.autograd.Variable(torch.rand(self.m * self.loads), requires_grad=True)

            # print(assignment_base.requires_grad)
            optimizer = torch.optim.Adam([assignment_base, assignment_priorities], lr=self.lr)

            tol = 5e-2
            best_obj = -np.inf
            ctr = 0
            num_epochs = int(10000)
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                loss, usw_contrib, se_contrib, cov_contrib, rev_contrib = \
                    self.compute_objective(assignment_base, assignment_priorities, u_k, u_r, u_p)
                loss = -1 * loss
                loss.backward()
                optimizer.step()

                obj = -1 * loss.item()
                # print("obj: ", obj)
                if obj < best_obj + tol:
                    ctr += 1
                else:
                    ctr = 0
                    best_obj = obj
                if ctr >= 100:
                    break
                if epoch % 10 == 0:
                    print("obj: ", obj)
                    print("usw contrib: ", usw_contrib)
                    print("se_contrib: ", se_contrib)
                    print("cov_contrib: ", cov_contrib)
                    print("rev_contrib: ", rev_contrib)

            if epoch == num_epochs:
                print("didn't break")

            base = assignment_base.data
            priorities = assignment_priorities.data
            allocation = self.assn_base_to_feasible_assn(base, priorities)

            # Update u_k using the subgradient. Note that strong envy is
            # clamped at 0 and lam_k is positive, so u_k will always be non-negative.
            strong_envy = self.compute_strong_envy(allocation)
            rev_dup_violations = self.compute_rev_dup_violations(allocation)
            paper_cov_violations = self.compute_paper_cov_violations(allocation)

            if torch.sum(strong_envy) < 1e-9 and \
                    torch.sum(rev_dup_violations) < 1e-9 and \
                    torch.sum(paper_cov_violations) < 1e-9:
                return self.convert_to_alloc(allocation.numpy())
            if torch.sum(strong_envy) > 1e-9:
                u_k += lam_k * strong_envy/torch.sum(strong_envy)
            if torch.sum(rev_dup_violations) > 1e-9:
                u_r += lam_k * rev_dup_violations/torch.sum(rev_dup_violations)
            if torch.sum(paper_cov_violations) > 1e-9:
                u_p += lam_k * paper_cov_violations/torch.sum(paper_cov_violations)

            # Update t_k
            if obj >= outer_prev_obj:
                counter += 1
            else:
                counter = 0
            if counter % outer_tol == 0:
                lam_k /= 2
            if counter == outer_tol**2:
                break

            print("Max USW - penalty: ", obj)
            print("USW: ", torch.sum(allocation * self.scores))
            print("Strong Envy (EF1): ", torch.sum(strong_envy))
            alloc_this_iter = self.convert_to_alloc(allocation.numpy())
            print(alloc_this_iter)
            print_stats(alloc_this_iter, self.scores[:self.m, :].numpy(), np.array([self.covs]*self.n))

        return self.convert_to_alloc(allocation.numpy())


def run_algo(dataset, base_dir):
    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)[0]
    paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)[0]

    sgd_searcher = SGDSearcher(paper_reviewer_affinities, reviewer_loads, paper_capacities)

    alloc = sgd_searcher.run_lagrangian_relaxation()
    return alloc


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = args.base_dir
    alloc_file = args.alloc_file
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # start = time.time()
    alloc = run_algo(dataset, base_dir)
    # runtime = time.time() - start

    # n = 3
    # m = 5
    # valuations = np.random.rand(m, n)
    # loads = 3
    # covs = 3
    #
    # sgd_searcher = SGDSearcher(valuations, loads, covs)
    #
    # alloc = sgd_searcher.run_lagrangian_relaxation()
    #
    # print(valuations)
    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, paper_capacities)
    # save_alloc(alloc, alloc_file)
    #
    # # print("Barman Algorithm Results")
    # # print("%.2f seconds" % runtime)
    # # print_stats(alloc, paper_reviewer_affinities, paper_capacities)
    #
    # print("SGD EFX Results")
    # print("%.2f seconds" % runtime)
    #
    # paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    # reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    # paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    #
    # print(alloc)
    # print_stats(alloc, paper_reviewer_affinities, paper_capacities)
