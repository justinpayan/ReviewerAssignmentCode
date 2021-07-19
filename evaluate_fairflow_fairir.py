import math
import networkx as nx
import pickle
import os

from utils import *


if __name__ == "__main__":
    # We want to obtain a 1.44-NSW allocation, which should also be 4-epsilon price envy free up to 1 item.
    args = parse_args()
    base_dir = args.base_dir
    dataset = args.dataset

    scores = np.load(os.path.join(base_dir, dataset, "scores.npy"))
    covs = np.load(os.path.join(base_dir, dataset, "covs.npy"))
    loads = np.load(os.path.join(base_dir, dataset, "loads.npy")).astype(np.int64)

    # Load the fairflow solution without the reviewer lower bounds...
    timestamps = {"cvpr": "2020-09-16-10-28-42", "cvpr2018": "2020-09-16-10-26-09", "midl": "2020-09-16-09-32-53"}

    fairflow_soln = load_fairflow_soln("/home/justinspayan/Fall_2020/fair-matching/exp_out/%s/fairflow/"
                                       "%s/results/assignment.npy" % (dataset, timestamps[dataset]))
    print("Fairflow Results")
    print_stats(fairflow_soln, scores, covs)
    print("***********\n***********\n")

    # FairIR solutions
    timestamps = {"cvpr": "2021-01-07-12-07-19", "cvpr2018": "2021-01-07-12-29-03", "midl": "2020-09-18-14-42-49"}

    fairir_soln = load_fairflow_soln("/home/justinspayan/Fall_2020/fair-matching/exp_out/%s/fairir/"
                                       "%s/results/assignment.npy" % (dataset, timestamps[dataset]))
    print("FairIR Results")
    print_stats(fairir_soln, scores, covs)
    print("***********\n***********\n")

