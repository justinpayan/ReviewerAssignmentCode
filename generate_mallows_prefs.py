from preflibtools import io, generate_profiles
from utils import *


if __name__ == "__main__":
    seed = 31415

    np.random.seed(seed)
    random.seed(seed)

    n = 10
    m = 20
    c = 3 # Num goods (revs) per agent (paper)
    U = 4 # Max num agents (papers) per good (rev)
    phis = [0, .25, .5, .75, .95]
    num_iters = 5

    for phi, iter in product(phis, range(num_iters)):
        cmap = generate_profiles.gen_cand_map(m)
        refm, refc = generate_profiles.gen_impartial_culture_strict(1, cmap)
        ref = io.rankmap_to_order(refm[0])
        rmaps, rmapscounts = generate_profiles.gen_mallows(n, cmap, [1], [phi], [ref])
        print(rmaps)
        print(rmapscounts)

        pra = np.zeros((m, n))
        loads = np.ones(m)*U
        covs = np.ones(n)*c

        for agent in range(n):
            if phi == 0:
                rmap = rmaps[0]
            else:
                rmap = rmaps[agent]
            for good in range(m):
                pra[good, agent] = rmap[good+1]

        np.save("mallows_scores_{}_{}.npy".format(phi, iter), pra)
        np.save("mallows_loads_{}_{}.npy".format(phi, iter), loads)
        np.save("mallows_covs_{}_{}.npy".format(phi, iter), covs)
