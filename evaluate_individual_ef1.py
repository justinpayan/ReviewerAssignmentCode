from absl import app
from absl import flags

import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('assignment', '', 'Location of the assignment to evaluate')
flags.DEFINE_string('affinities', '', 'Paper reviewer affinity scores')

# Example run:
# python evaluate_individual_ef1.py \
# --assignment exp_out/midl/fairflow/2020-09-16-09-32-53/results/assignment.npy \
# --affinities data/midl/scores.npy


def evaluate_ef1(assignment_filename, pr_affs_filename):
    A = np.load(assignment_filename)
    S = np.load(pr_affs_filename)
    num_papers = S.shape[1]

    # Compute utility
    def u(paper):
        return np.sum(A[:, paper] * S[:, paper])

    # If we can drop one non_zero element of the other paper's allocation and still prefer it, return True
    # Else return False
    def check_envy_to_1(u, p, op):
        non_zero_idxs = np.where(A[:, op] > 0)[0]
        for i in non_zero_idxs:
            reduced_bundle = np.copy(A[:, op])
            reduced_bundle[i] -= 1
            reduced_utility = np.sum(reduced_bundle * S[:, p])
            if reduced_utility > u:
                print("Paper %d envies paper %d, even after reviewer %d is removed" % (p, op, i))
                return True
        return False

    for paper in range(num_papers):
        print(paper)
        utility = u(paper)
        for other_paper in range(num_papers):
            if paper != other_paper:
                if check_envy_to_1(utility, paper, other_paper):
                    print("Not EF1 - Allocation is NOT EF1 for individual papers")
                    return
    print("Yes EF1 - Allocation is EF1 for individual papers")


def main(argv):
    evaluate_ef1(FLAGS.assignment, FLAGS.affinities)


if __name__ == '__main__':
    app.run(main)
