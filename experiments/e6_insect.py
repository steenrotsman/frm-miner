from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from mass_ts import mass2 as mass
from plot import plot_motifs, remove_spines
from scipy.io import loadmat
from scipy.stats import zscore
from stumpy import ostinato

from frm import Miner

MINSUP = [1, 0.75, 0.5]
SEGLEN = 30
ALPHABET = 4

USE_PRECOMPUTED = True


def main():
    insect = loadmat("e6_insectAllData.mat")
    data = [np.reshape(insect[f"d{i}{i}"], (-1)) for i in range(1, 5)]
    plot_data(data)
    plot_motif(data)


def plot_data(data):
    fig, axs = plt.subplots(ncols=3)

    # Plot data
    for row in data:
        axs[0].plot(row.T, lw=0.3)

    # Find and plot consensus motif
    _, ts_index, ss_index = (
        (8.070564764776059 + 1.059678327857585e-12j),
        2,
        96423,
    )
    if not USE_PRECOMPUTED:
        _, ts_index, ss_index = ostinato(data, 800)

    # Aesthetics
    axs[0].set(xticks=[0, 120000])
    remove_spines(axs[0], remove_y=False)
    axs[1].plot(zscore(data[ts_index][ss_index : ss_index + 800]), color="k", lw=0.3)
    axs[1].set(ylim=(-3, 3), xticks=[0, 800], yticks=[-2, 0, 2])
    remove_spines(axs[1], remove_y=False)

    for i, row in enumerate(data):
        if i == ts_index:
            start = ss_index
        else:
            start = np.argmin(mass(row, data[ts_index][ss_index : ss_index + 800]))
        axs[2].plot(zscore(row[start : start + 800]), lw=0.3)
        axs[2].set(ylim=(-3, 3), xticks=[0, 800], yticks=[-2, 0, 2])

    remove_spines(axs[2], remove_y=False)

    plt.savefig(join("figs", "6 insect data.eps"))
    plt.savefig(join("figs", "6 insect data.png"))
    plt.close()


def plot_motif(data):
    fig, axs = plt.subplots(ncols=len(MINSUP), sharey="all")
    for minsup, ax in zip(MINSUP, axs):
        # Mine frequent motifs in differenced data
        miner = Miner(minsup, SEGLEN, ALPHABET, diff=1)
        motifs = miner.mine(data)
        motif = motifs[0]
        plot_motifs([ax], data, [motif])
        ax.set(ylim=(-3, 3), xlabel=f"minsup = {minsup}", yticks=[-2, 0, 2])
    plt.savefig(join("figs", "6 insect motifs.eps"))
    plt.savefig(join("figs", "6 insect motifs.png"))
    plt.close()


if __name__ == "__main__":
    main()
