from utils import *


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    dataset = args.dataset
    fair_matching_dir = args.fair_matching_dir
    fairflow_timestamp = args.fairflow_timestamp
    fairir_timestamp = args.fairir_timestamp

    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)

    fairflow_soln = load_fairflow_soln("%s/exp_out/%s/fairflow/%s/results/assignment.npy" %
                                       (fair_matching_dir, dataset, fairflow_timestamp))
    print("Fairflow Results")
    print_stats(fairflow_soln, scores, covs)
    print("***********\n***********\n")

    fairir_soln = load_fairflow_soln("%s/exp_out/%s/fairir/%s/results/assignment.npy" %
                                     (fair_matching_dir, dataset, fairir_timestamp))
    print("FairIR Results")
    print_stats(fairir_soln, scores, covs)
    print("***********\n***********\n")

