from autoassigner import *
from utils import *


def pr4a(pra, covs, loads, iter_limit):
    # Normalize the affinities so they're between 0 and 1
    pra[np.where(pra < 0)] = 0
    pra /= np.max(pra)

    pr4a_instance = auto_assigner(pra, demand=covs[0], ability=loads, iter_limit=iter_limit)
    pr4a_instance.fair_assignment()

    alloc = pr4a_instance.fa

    return alloc


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    dataset = args.dataset
    alloc_file = args.alloc_file

    iter_limit = np.inf
    if dataset != "midl":
        iter_limit = 1

    paper_reviewer_affinities = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)

    alloc = pr4a(paper_reviewer_affinities, covs, loads, iter_limit)

    save_alloc(alloc, alloc_file)

    print(alloc)
    print_stats(alloc, paper_reviewer_affinities, covs)
