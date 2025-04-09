from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import stumpy
from plot import remove_spines
from scipy.stats import zscore

from frm import Miner

# Simulation settings
UNITS = 100
TS_LEN = 10000
MOTIF_LEN = 512
INJECT = 75
MOTIF = zscore(np.sin(np.linspace(0, 10, MOTIF_LEN)))
NOISE_LEVELS = [0.4, 0.6, 0.8, 1.0, 1.2]
PRECOMPUTED = [(36, 7595), (22, 4855), (79, 1579), (26, 6244), (13, 7735)]

# Parameters
MINSUP = 0.3
SEGLEN = 25
ALPHA = 4
K = 1

RNG = np.random.default_rng(1234)
PLOT = 3


def main():
    # Plot example data
    data = get_data(0.8)
    fig, axs = plt.subplots(nrows=PLOT, sharex="all")
    for row, ax in zip(data, axs):
        ax.plot(row, "k", lw=0.3)
        ax.set(yticks=[-2.5, 2.5])
        remove_spines(ax, remove_y=False)
    plt.savefig(join("figs", "3 data.png"))
    plt.savefig(join("figs", "3 data.eps"))
    plt.close()

    # Vary noise levels
    fix, axs = plt.subplots(ncols=len(NOISE_LEVELS), sharey="all")
    for level, ax in zip(NOISE_LEVELS, axs):
        print("Discovering motifs for noise level", level)
        data = get_data(level)

        # Discover motifs
        mm = Miner(MINSUP, SEGLEN, ALPHA, k=K, diff=1)
        motifs = mm.mine(data)

        # Plot motif
        motif = motifs[0]
        for ts, idx in motif.best_matches.items():
            ax.plot(zscore(data[ts][idx : idx + motif.length]), lw=0.3)
        ax.set(xticks=[0, motif.length], xlabel=level)
        remove_spines(ax, remove_y=False)
    plt.savefig(join("figs", "3 accuracy.png"))
    plt.savefig(join("figs", "3 accuracy.eps"))
    plt.close()

    # Discover with Ostinato
    fig, axs = plt.subplots(ncols=len(NOISE_LEVELS), sharey="all")
    for level, ax, precomputed in zip(NOISE_LEVELS, axs, PRECOMPUTED):
        data = get_data(level)
        consensus_motif = ostinato(data, MOTIF_LEN, precomputed)
        ax.plot(zscore(consensus_motif), "k", lw=0.3)
        ax.set(xticks=[0, MOTIF_LEN])
        remove_spines(ax, remove_y=False)
    plt.savefig(join("figs", "3 consensus.png"))
    plt.savefig(join("figs", "3 consensus.eps"))
    plt.close()


def get_data(noise_level):
    noise = RNG.normal(size=(UNITS, TS_LEN), scale=3)
    locations = RNG.integers(0, TS_LEN - MOTIF_LEN, size=len(noise))
    locations[INJECT:] = -1
    RNG.shuffle(locations)
    for i, loc in enumerate(locations):
        if loc == -1:
            continue
        else:
            # Inject motif
            noisy_motif = MOTIF + RNG.normal(size=MOTIF_LEN, scale=noise_level)
            noise[i][loc : loc + MOTIF_LEN] = noisy_motif

    return zscore(np.cumsum(noise, axis=1), axis=1)


def ostinato(data, m, precomputed=None):
    if precomputed is None:
        radius, idx, subidx = stumpy.ostinato(data, m)
        print(idx, subidx)
    else:
        idx, subidx = precomputed
    return data[idx][subidx : subidx + m]


if __name__ == "__main__":
    main()
