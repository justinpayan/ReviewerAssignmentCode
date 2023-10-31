import math
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    runtimes_tpms_ijcai = [1405.4896247386932, 1400.4237909317017, 1202.5629422664642, 1463.0609667301178,
                           1272.8446960449219, 1070.535500049591, 1069.0648872852325, 1026.2036473751068,
                           1057.303522348404, 1058.185992717743]
    runtimes_fairflow_cvpr = [965.2493062019348, 896.4305896759033, 957.9317746162415, 927.1088373661041,
                              983.8674857616425, 940.9907424449921, 943.0574164390564, 935.6805827617645,
                              928.2986760139465, 857.0464789867401]
    runtimes_fairflow_cvpr2018 = [3143.492673635483, 2933.8753592967987, 3111.8740389347076, 3018.8485159873962,
                                  3188.989809989929, 3069.98836517334, 3073.389800786972,
                                  3050.3381712436676, 3022.679287672043, 2789.984931707382]
    runtimes_fairseq_ijcai = [69.5730242729187, 61.02978730201721, 63.61751651763916, 66.09009718894958,
                              69.67499494552612, 63.09398150444031, 63.55621790885925, 63.48904728889465,
                              69.26059484481812, 57.468377351760864]
    runtimes_fairflow_ijcai = [2218.8225157260895, 2041.329702615738, 2117.664055109024, 2118.1779487133026,
                               2253.327534198761, 2156.5529425144196, 2165.3746948242188, 2140.6092710494995,
                               2123.266214132309, 1926.245224237442]

    # runtime_means = {"MIDL":
    #                      {"FairFlow": 2.22, "TPMS": 0.96, "PR4A": 0.00, "FairSequence": 0.22},
    #                  "CVPR":
    #                      {"FairFlow": 853.36, "TPMS": 228.98, "PR4A": 3491.30, "FairSequence": 56.82},
    #                  "CVPR 2018":
    #                      {"FairFlow": 2838.79, "TPMS": 1050.79, "PR4A": 12129.82, "FairSequence": 299.62}}
    # runtime_stds = {"MIDL":
    #                     {"FairFlow": 0.06, "TPMS": 0.01, "PR4A": 0.00, "FairSequence": 0.01},
    #                 "CVPR":
    #                     {"FairFlow": 2.96, "TPMS": 2.30, "PR4A": 470.63, "FairSequence": 0.40},
    #                 "CVPR 2018":
    #                     {"FairFlow": 110.54, "TPMS": 17.21, "PR4A": 329.82, "FairSequence": 1.15}}
    runtime_means = {"CVPR":
                         {"FairFlow": 853.36, "TPMS": 228.98, "PR4A": 3491.30, "FairSequence": 56.82},
                     "CVPR 2018":
                         {"FairFlow": 2838.79, "TPMS": 1050.79, "PR4A": 12129.82, "FairSequence": 299.62}}
    runtime_stds = {"CVPR":
                        {"FairFlow": 2.96, "TPMS": 2.30, "PR4A": 470.63, "FairSequence": 0.40},
                    "CVPR 2018":
                        {"FairFlow": 110.54, "TPMS": 17.21, "PR4A": 329.82, "FairSequence": 1.15}}

    runtime_means["CVPR"]["FairFlow"] = np.mean(runtimes_fairflow_cvpr)
    runtime_stds["CVPR"]["FairFlow"] = np.std(runtimes_fairflow_cvpr)
    runtime_means["CVPR 2018"]["FairFlow"] = np.mean(runtimes_fairflow_cvpr2018)
    runtime_stds["CVPR 2018"]["FairFlow"] = np.std(runtimes_fairflow_cvpr2018)
    runtime_means["IJCAI"]["FairFlow"] = np.mean(runtimes_fairflow_ijcai)
    runtime_stds["IJCAI"]["FairFlow"] = np.std(runtimes_fairflow_ijcai)
    runtime_means["IJCAI"]["TPMS"] = np.mean(runtimes_tpms_ijcai)
    runtime_stds["IJCAI"]["TPMS"] = np.std(runtimes_tpms_ijcai)
    runtime_means["IJCAI"]["FairSequence"] = np.mean(runtimes_fairseq_ijcai)
    runtime_stds["IJCAI"]["FairSequence"] = np.std(runtimes_fairseq_ijcai)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    algs = ["PR4A", "FairFlow", "TPMS", "FairSequence"]
    dsets = ["CVPR", "CVPR 2018", "IJCAI"]
    colors = ["blue", "brown", "black", "green"]
    markers = ["o", "d", "^", "*"]

    for j in range(3):
        for i in range(4):
            ax.errorbar(j, runtime_means[dsets[j]][algs[i]],
                        yerr=runtime_stds[dsets[j]][algs[i]],
                        color=colors[i],
                        marker=markers[i],
                        ecolor="black",
                        capsize=5,
                        capthick=1,
                        label=algs[i],
                        ls=""
                        )

    xlims = (-1, 5)
    ylims = (-5, 15)
    bar_ylims = (0, 15)

    xlabel = 'Dataset'
    ylabel = 'Runtime (s)'
    xticks = [0, 1, 2]
    xticklabels = dsets

    # for i, axes in enumerate(ax.flat):
    #     yticks = np.linspace(axes.get_ylim()[0], axes.get_ylim()[1], 5)
    #     yticklabels = yticks
    #     stylize_axes(axes, titles[i], xlabel, ylabel, xticks, yticks, xticklabels, yticklabels)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(top=False, direction='out', width=1)
    ax.yaxis.set_tick_params(right=False, direction='out', width=1)
    ax.set_title("Algorithm Runtimes")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax.legend(handles[:4], labels[:4])

    fig.tight_layout()
    fig.savefig('FairSequenceCompareRuntimes.png', dpi=300, bbox_inches='tight', transparent=True)
