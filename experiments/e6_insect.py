import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
import matplotlib.pyplot as plt
from itertools import product
from os import listdir
from multiprocessing import Pool

from frm._frm_py.miner import Miner
from ostinato import ostinato
from plot import remove_spines, COLORS

MINSUP = [1, 0.5]
SEGLEN = 50
ALPHABET = 4



def main():
    insect = loadmat('e6_insectAllData.mat')
    data = [np.reshape(insect[f'd{i}{i}'], (-1)) for i in range(1, 5)]
    plot_data(data)
    plot_motif(data)


def plot_data(data):
    fig, axs = plt.subplots(ncols=2, layout='constrained')

    # Plot data
    for row, color in zip(data, COLORS):
        axs[0].plot(row.T, color=color, lw=0.3)

    # Find and plot consensus motif
    # bsf_rad, ts_index, ss_index = ostinato(data, 800)
    bsf_rad, ts_index, ss_index = ((8.070564764776059+1.059678327857585e-12j), 2, 96423)

    # Aesthetics
    axs[0].set(xticks=[0, 120000], yticks=[])
    remove_spines(axs[0])
    axs[1].set(xticks=[0, 800], yticks=[])
    axs[1].plot(data[ts_index][ss_index:ss_index+800], color='k', lw=0.3)
    remove_spines(axs[1])

    plt.savefig(f'figs/6 insect data.eps')
    plt.close()


def plot_motif(data):
    fig, axs = plt.subplots(ncols=2, layout='constrained')
    diff = [np.diff(row) for row in data]

    for minsup, ax in zip(MINSUP, axs):
        # Mine frequent motifs in differenced data
        miner = Miner(minsup, SEGLEN, ALPHABET, k=1)
        motifs = miner.mine(diff)
        motif = motifs[0]

        # Apply frequent motif indexes to z-normalised data
        representative = np.mean([zscore(data[ts][idx : idx+motif.length]) for ts, idx in motif.best_matches.items()], axis=0)
        ax.plot(representative, color='b', lw=0.5)
        for ts, idx in motif.best_matches.items():
            ax.plot(zscore(data[ts][idx : idx+motif.length]), color='k', lw=0.1)
        ax.set(ylim=(-3, 3), xticks=[0, len(representative)], yticks=[])
        remove_spines(ax)
    plt.savefig(f'figs/6 insect motifs.eps')
    plt.close()


if __name__ == '__main__':
    main()