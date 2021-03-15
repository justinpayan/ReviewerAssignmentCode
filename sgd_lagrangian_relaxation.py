import os
import random
import time
import torch

from torch import nn
from utils import *


class SGDSearcher(object):
    def __init__(self, scores, loads, covs):
        self.scores = torch.tensor(scores).type(torch.FloatTensor)
        self.loads = loads
        self.covs = covs
        self.m, self.n = scores.shape
        self.lr = 3e-2
        self.diag_mask = torch.ones((self.n, self.n)) - torch.eye(self.n)

    def convert_to_alloc(self, matrix_alloc):
        alloc = {i: list() for i in range(self.n)}
        agents_getting_each_item = np.argmax(matrix_alloc, axis=1)
        for g in range(self.m):
            alloc[agents_getting_each_item[g]].append(g)
        return alloc

    def compute_strong_envy(self, allocation):
        # print(allocation.type())
        # print(self.scores.type())
        # The ij element should be i's valuation for j's bundle
        W = torch.matmul(torch.transpose(self.scores, 0, 1), allocation)
        # I want n matrices, where each matrix i has rows j which hold the value of i for items given to j.
        # M = (self.scores.reshape((1, self.m, self.n)) * allocation.reshape((self.m, 1, self.n))).transpose(0, 1)
        # M = torch.max(M, dim=2).values
        M = self.scores.transpose(0, 1).reshape(self.n, self.m, 1) * allocation.reshape(1, self.m, self.n)
        M = torch.max(M, dim=1).values
        # M's ij element is most valuable item for i in j's bundle. If we just change it so that all items not owned
        # by j are worth 1000 and then take the min, this will be EFX instead of EF1.
        strong_envy = (W - M) - torch.diag(W).reshape((-1, 1))
        strong_envy = torch.clamp(strong_envy, min=0)
        strong_envy = strong_envy * self.diag_mask
        return strong_envy

    def compute_objective(self, x_k, u_k):
        # Convert x_k into a randomized allocation using just softmax
        # Or maybe we should convert x_k into a deterministic allocation using gumbel softmax... let's see.
        # If we do gumbel softmax, I think that would mean we are maximizing the expected value of the objective.
        # That could work... but first just see if using softmax alone/softmax plus a penalty on entropy will give
        # a deterministic allocation in the end.
        allocation = nn.functional.softmax(x_k, dim=1)
        # allocation = nn.functional.gumbel_softmax(x_k, dim=1, hard=True)
        usw = torch.sum(allocation * self.scores)

        # Compute the penalty based on strong envy and the lagrange multiplier for each envy comparison
        strong_envy = self.compute_strong_envy(allocation)
        penalty = torch.sum(u_k * strong_envy)
        return usw - penalty

    def run_lagrangian_relaxation(self):
        lam_k = 2
        u_k = torch.ones((self.n, self.n)) * self.diag_mask

        allocation = None

        outer_tol = 5
        outer_prev_obj = np.inf
        counter = 0
        for outer_iter in range(1000):
            # Find the x_k that maximizes the penalized USW
            assignment_base = torch.autograd.Variable(torch.ones(self.scores.shape), requires_grad=True)
            # print(assignment_base.requires_grad)
            optimizer = torch.optim.Adam([assignment_base], lr=self.lr)

            tol = 1e-6
            prev_obj = -np.inf
            ctr = 0
            for epoch in range(int(1e20)):
                optimizer.zero_grad()
                loss = -1*self.compute_objective(assignment_base, u_k)
                loss.backward()
                optimizer.step()

                obj = -1 * loss.item()
                # print("obj: ", obj)
                if obj - prev_obj < tol:
                    ctr += 1
                else:
                    ctr = 0
                if ctr >= 10:
                    break
                prev_obj = obj

            if epoch == 1e8:
                print("didn't break")

            assignment_this_iter = assignment_base.data
            allocation = nn.functional.softmax(assignment_this_iter, dim=1)

            # Update u_k using the subgradient. Note that strong envy is
            # clamped at 0 and lam_k is positive, so u_k will always be non-negative.
            strong_envy = self.compute_strong_envy(allocation)

            if torch.sum(strong_envy) < 1e-9:
                return self.convert_to_alloc(allocation.numpy())

            u_k += lam_k * strong_envy/torch.sum(strong_envy)

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

        return self.convert_to_alloc(allocation.numpy())


def run_algo(dataset, base_dir):
    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    reviewer_loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    paper_capacities = np.load(os.path.join(base_dir, dataset, "covs.npy")).astype(np.int64)

    sgd_searcher = SGDSearcher(paper_reviewer_affinities, reviewer_loads, paper_capacities)

    alloc = sgd_searcher.run_lagrangian_relaxation()
    return alloc


if __name__ == "__main__":
    # args = parse_args()
    # dataset = args.dataset
    # base_dir = args.base_dir
    # alloc_file = args.alloc_file
    # initial_order = args.local_search_init_order
    # num_processes = args.num_processes

    seed = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # start = time.time()
    # alloc = run_algo(dataset, base_dir)
    # runtime = time.time() - start

    n = 3
    m = 9
    valuations = np.random.rand(m, n)
    loads = np.ones(m)
    covs = np.ones(m//n)

    sgd_searcher = SGDSearcher(valuations, loads, covs)

    alloc = sgd_searcher.run_lagrangian_relaxation()

    print(valuations)
    print(alloc)

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
