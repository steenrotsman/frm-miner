from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from plot import remove_spines
from scipy.io import loadmat
from scipy.stats import zscore
from stumpy import ostinato, stump

from frm import Miner

MINSUP = [1, 0.5]
SEGLEN = 30
ALPHABET = 4

USE_PRECOMPUTED = True


def main():
    insect = loadmat('e6_insectAllData.mat')
    data = [np.reshape(insect[f'd{i}{i}'], (-1)) for i in range(1, 5)]
    plot_data(data)
    plot_motif(data)
    plot_kofp(data)


def plot_data(data):
    fig, axs = plt.subplots(ncols=2)

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
    axs[1].plot(zscore(data[ts_index][ss_index : ss_index + 800]), color='k', lw=0.3)
    axs[1].set(ylim=(-3, 3), xticks=[0, 800], yticks=[-2, 0, 2])
    remove_spines(axs[1], remove_y=False)

    plt.savefig(join('figs', '6 insect data.eps'))
    plt.savefig(join('figs', '6 insect data.png'))
    plt.close()


def plot_motif(data):
    fig, axs = plt.subplots(ncols=2, layout='constrained', sharey='all')
    diff = [np.diff(row) for row in data]

    for minsup, ax in zip(MINSUP, axs):
        # Mine frequent motifs in differenced data
        miner = Miner(minsup, SEGLEN, ALPHABET)
        motifs = miner.mine(diff)
        motif = motifs[0]

        # Apply frequent motif indexes to z-normalised data
        representative = np.mean(
            [
                zscore(data[ts][idx : idx + motif.length])
                for ts, idx in motif.best_matches.items()
            ],
            axis=0,
        )
        ax.plot(representative, color='b', lw=1)
        for ts, idx in motif.best_matches.items():
            ax.plot(zscore(data[ts][idx : idx + motif.length]), color='k', lw=0.1)
        ax.set(ylim=(-3, 3), xticks=[0, len(representative)])
        remove_spines(ax, remove_y=False)
    plt.savefig(join('figs', '6 insect motifs.eps'))
    plt.savefig(join('figs', '6 insect motifs.png'))
    plt.close()


def plot_kofp(data):
    fig, axs = plt.subplots(ncols=3, sharey='all')
    bests = [[0, 65048, 1, 41809], [0, 65079, 1, 51051], [0, 65025, 1, 50989]]
    if not USE_PRECOMPUTED:
        for m, ax in zip((1100, 1140, 1200), axs):
            outs = []
            # As k=2, this is equal to the best motif pair across all ABJoins
            for i in range(4):
                # Though ABJoin(A,B)!=(ABJoin(B,A)), bestPair(A,B)==bestPair(B,A)
                for j in range(i + 1, 4):
                    if i == j:
                        continue
                    out = stump(data[i], m, data[j], ignore_trivial=False)
                    outs.append(
                        [i, np.argmin(out[:, 0]), j, out[np.argmin(out[:, 0])][1]]
                    )
            best = min(outs, key=lambda x: x[0])
            bests.append(best)
            print(best)
    for m, ax, best in zip((1100, 1140, 1200), axs, bests):
        ax.plot(zscore(data[best[0]][best[1] : best[1] + m]), lw=1, c='b')
        ax.plot(zscore(data[best[2]][best[3] : best[3] + m]), lw=0.3, c='k')
        ax.set(ylim=(-3, 3), xticks=[0, m], yticks=[-2, 0, 2])
        remove_spines(ax, remove_y=False)
    plt.savefig(join('figs', '6 insect kofP.eps'))
    plt.savefig(join('figs', '6 insect kofP.png'))
    plt.close()


if __name__ == '__main__':
    main()
