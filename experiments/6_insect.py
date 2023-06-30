import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
import matplotlib.pyplot as plt

from frm._frm_py.miner import Miner
from ostinato import ostinato
from plot import remove_spines, COLORS


def main():
    insect = loadmat('6_insectAllData.mat')
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

    plt.savefig(f'figs/6 insect data')


def plot_motif(data):
    fig, ax = plt.subplots(layout='constrained')

    # Mine frequent motifs in differenced data
    diff = [np.diff(row) for row in data]
    miner = Miner(0.75, 10, 9, k=1, min_len=80, max_overlap=1)
    motifs = miner.mine_motifs(diff)

    for i, motif in enumerate(motifs):
        # Apply frequent motif indexes to z-normalised data
        representative = np.mean([zscore(data[ts][idxs[0] : idxs[1]]) for ts, idxs in motif.match_indexes.items()], axis=0)
        ax.plot(representative, color='b', lw=0.5)
        for ts, idxs in motif.match_indexes.items():
            ax.plot(zscore(data[ts][idxs[0] : idxs[1]]), color='k', lw=0.1)
        ax.set(ylim=(-3, 3), xticks=[0, len(representative)], yticks=[])
        remove_spines(ax)
    plt.savefig(f'figs/6 insect motif')


if __name__ == '__main__':
    main()