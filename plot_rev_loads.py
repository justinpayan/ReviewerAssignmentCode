import matplotlib.pyplot as plt

data = {'FairFlow':
            {'MIDL': [(0, 69), (1, 13), (2, 12), (3, 15), (4, 68)],
             'CVPR': [(0, 1), (1, 0), (2, 20), (3, 58), (4, 33), (5, 43), (6, 1218)],
             'CVPR2018': [(0, 206), (1, 154), (2, 176), (3, 256), (4, 479), (5, 124), (6, 186), (7, 449), (8, 173),
                          (9, 637)],
             },
        'FairIR':
            {'MIDL': [(0, 69), (1, 13), (2, 11), (3, 17), (4, 67)],
             'CVPR': [(0, 3), (1, 5), (2, 12), (3, 39), (4, 43), (5, 75), (6, 1196)],
             'CVPR2018': [(0, 221), (1, 162), (2, 154), (3, 236), (4, 538), (5, 98), (6, 164), (7, 444), (8, 133), (9, 690)],
             },
        'TPMS':
            {'MIDL': [(0, 69), (1, 13), (2, 12), (3, 15), (4, 68)],
             'CVPR': [(0, 1), (1, 3), (2, 18), (3, 26), (4, 52), (5, 94), (6, 1179)],
             'CVPR2018': [(0, 227), (1, 161), (2, 156), (3, 231), (4, 538), (5, 103), (6, 149), (7, 447), (8, 122), (9, 706)],
             },
        'PR4A':
            {'MIDL': [(0, 69), (1, 14), (2, 13), (3, 10), (4, 71)],
             'CVPR': [(0, 61), (1, 0), (2, 0), (3, 1), (4, 0), (5, 0), (6, 1311)],
             'CVPR2018': [(0, 228), (1, 163), (2, 159), (3, 222), (4, 531), (5, 110), (6, 143), (7, 459), (8, 131), (9, 694)]
             },
        'GRRR':
            {'MIDL': [(0, 69), (1, 17), (2, 7), (3, 13), (4, 71)],
             'CVPR': [(0, 52), (1, 9), (2, 2), (3, 1), (4, 0), (5, 1), (6, 1308)],
             'CVPR2018': [(0, 234), (1, 151), (2, 161), (3, 233), (4, 525), (5, 102), (6, 160), (7, 454), (8, 114), (9, 706)]
             }
        }

for conference in ["MIDL", "CVPR", "CVPR2018"]:
    plt.clf()
    offset = 0
    w = 1
    for algorithm in data:
        plt.bar([x[0]*10+offset for x in data[algorithm][conference]],
                [x[1] for x in data[algorithm][conference]],
                width=w,
                label=algorithm)
        plt.xticks(ticks=[x[0]*10+2*w for x in data[algorithm][conference]],
                   labels=[str(x[0]) for x in data[algorithm][conference]])
        offset += w
    plt.legend()
    plt.title("Reviewer Load Distribution (%s)" % conference)
    plt.xlabel("Reviewing Load")
    plt.ylabel("Number of Reviewers")
    plt.savefig("review_load_dist_%s.png" % conference)