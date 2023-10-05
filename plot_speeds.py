import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    runtime_means = {"MIDL":
                         {"FairFlow": 2.22, "TPMS": 0.96, "PR4A": 0.00, "FairSequence": 0.22},
                     "CVPR":
                         {"FairFlow": 853.36, "TPMS": 228.98, "PR4A": 3491.30, "FairSequence": 56.82},
                     "CVPR 2018":
                         {"FairFlow": 2838.79, "TPMS": 1050.79, "PR4A": 12129.82, "FairSequence": 299.62}}
    runtime_stds = {"MIDL":
                        {"FairFlow": 0.06, "TPMS": 0.01, "PR4A": 0.00, "FairSequence": 0.01},
                    "CVPR":
                        {"FairFlow": 2.96, "TPMS": 2.30, "PR4A": 470.63, "FairSequence": 0.40},
                    "CVPR 2018":
                        {"FairFlow": 110.54, "TPMS": 17.21, "PR4A": 329.82, "FairSequence": 1.15}}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    algs = ["PR4A", "FairFlow", "TPMS", "FairSequence"]
    dsets = ["MIDL", "CVPR", "CVPR 2018"]
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
