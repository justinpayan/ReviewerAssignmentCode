import sys

from utils import *


def estimate_gamma(pra, covs, loads, best_revs, alpha):
    m, n = pra.shape

    gamma_values = []

    sizes = [2]
    interval = n // 20
    sizes.extend(list(range(interval, n, interval)))
    sizes.append(n)
    print(sizes[:3])
    print(sizes[-3:])

    for _ in range(5):
        for s in sizes:
            if len(gamma_values):
                print(s, max(gamma_values))
            else:
                print(s)

            num_agents_Ye = np.random.randint(3, n+1)
            _agents_Ye = list(random.sample(range(n), num_agents_Ye))
            bin_agents_Ye = np.zeros(n)
            bin_agents_Ye[_agents_Ye] = 1
            e = np.random.choice(np.where(bin_agents_Ye)[0])
            bin_agents_Y = bin_agents_Ye.copy()
            bin_agents_Y[e] = 0
            agents_Ye = np.where(bin_agents_Ye)[0].tolist()
            agents_Y = np.where(bin_agents_Y)[0].tolist()

            num_agents_X = np.random.randint(1, num_agents_Ye-1)
            _agents_X = list(random.sample(agents_Y, num_agents_X))
            bin_agents_X = np.zeros(n)
            bin_agents_X[_agents_X] = 1
            bin_agents_Xe = bin_agents_X.copy()
            bin_agents_Xe[e] = 1
            agents_Xe = np.where(bin_agents_Xe)[0].tolist()
            agents_X = np.where(bin_agents_X)[0].tolist()

            positions_Ye = sorted(range(n), key=lambda x: random.random())[:num_agents_Ye]
            position_map = dict(zip(sorted(agents_Ye), positions_Ye))

            X = {(a, position_map[a]) for a in agents_X}
            Xe = {(a, position_map[a]) for a in agents_Xe}
            Y = {(a, position_map[a]) for a in agents_Y}
            Ye = {(a, position_map[a]) for a in agents_Ye}

            vals = []

            for inp in [X, Y, Xe, Ye]:
                seln_order = [x[0] for x in sorted(inp, key=lambda x: x[1])]
                vals.append(safe_rr_usw(seln_order, pra, covs, loads, best_revs)[0]*(len(inp)**alpha))

            diff_lhs = vals[2] - vals[0]
            diff_rhs = vals[3] - vals[1]

            if diff_lhs < 0 or diff_rhs < 0:
                print("NOT MONOTONIC")
                sys.exit(0)

            if 0 <= diff_lhs < diff_rhs and not np.isclose(diff_lhs, diff_rhs):
                gamma_values.append(diff_rhs/diff_lhs)
    print("final ", max(gamma_values))


def estimate_alpha(pra, covs, loads, best_revs, min_size=5):
    m, n = pra.shape

    alpha_values = []
    num_nonmonotonic = 0

    all_tuples = list(product(range(n), range(n)))

    sizes = [2]
    interval = n//20
    sizes.extend(list(range(interval, n, interval)))
    sizes.append(n)

    for _ in range(5):
        for s in sizes:
            if len(alpha_values):
                print(s, num_nonmonotonic, max(alpha_values))
            else:
                print(s, num_nonmonotonic)

            Xe = random.sample(range(n), s)
            positions_Xe = sorted(range(n), key=lambda x: random.random())[:s]
            position_map = dict(zip(sorted(Xe), positions_Xe))
            tuples_Xe = {(a, position_map[a]) for a in Xe}
            tuples_X = random.sample(tuples_Xe, s-1)

            vals = []
            for inp in [tuples_X, tuples_Xe]:
                seln_order = [x[0] for x in sorted(inp, key=lambda x: (x[1], x[0]))]
                dedup = []
                for p in seln_order:
                    if p not in dedup:
                        dedup.append(p)
                vals.append(safe_rr_usw(dedup, pra, covs, loads, best_revs)[0])

            if vals[1] < vals[0]:
                alpha_values.append(math.log(vals[1] / vals[0]) / math.log(1.0 * len(tuples_X) / len(tuples_Xe)))
                num_nonmonotonic += 1

    print("final ", num_nonmonotonic, max(alpha_values))


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = args.base_dir
    alloc_file = args.alloc_file

    random.seed(args.seed)

    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))

    best_revs = np.argsort(-1 * scores, axis=0)

    # estimate_alpha(scores, covs, loads, best_revs)
    alphas = {"midl": .01, "cvpr": 1.03, "cvpr2018": .505}
    estimate_gamma(scores, covs, loads, best_revs, alphas[dataset])
