from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import stumpy
from frm import Miner
from scipy.stats import zscore

from plot import remove_spines

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
    data, locations = get_data(0.8, RNG)
    fig, ax = plt.subplots()
    for row, start in zip(data[:PLOT], locations):
        end = start + MOTIF_LEN
        ax.plot(range(start), row[:start], "k", lw=0.3)
        ax.plot(range(start, end), row[start:end], "b", lw=0.3)
        ax.plot(range(end, len(row)), row[end:], "k", lw=0.3)
        ax.set(yticks=[-2.5, 2.5])
        remove_spines(ax, remove_y=False)
    plt.savefig(join("figs", "Fig12.pdf"))
    plt.close()

    # Vary noise levels
    fix, axs = plt.subplots(ncols=len(NOISE_LEVELS), sharey="all")
    for level, ax in zip(NOISE_LEVELS, axs):
        print("Discovering motifs for noise level", level)
        data, _ = get_data(level, RNG)

        # Discover motifs
        mm = Miner(MINSUP, SEGLEN, ALPHA, k=K, diff=1)
        motifs = mm.mine(data)

        # Plot motif
        motif = motifs[0]
        for ts, idx in motif.best_matches.items():
            ax.plot(zscore(data[ts][idx : idx + motif.length]), lw=0.3)
        ax.set(xticks=[0, motif.length], xlabel=level)
        remove_spines(ax, remove_y=False)
    plt.savefig(join("figs", "Fig13.pdf"))
    plt.close()

    # Discover with Ostinato
    fig, axs = plt.subplots(ncols=len(NOISE_LEVELS), sharey="all")
    for level, ax, precomputed in zip(NOISE_LEVELS, axs, PRECOMPUTED):
        data, locations = get_data(level, RNG)
        consensus_motif = ostinato(data, MOTIF_LEN, precomputed)
        ax.plot(zscore(consensus_motif), "k", lw=0.3)
        ax.set(xticks=[0, MOTIF_LEN], xlabel=level)
        remove_spines(ax, remove_y=False)
    plt.savefig(join("figs", "Fig14.pdf"))
    plt.close()


def get_data(noise_level, rng):
    noise = rng.normal(size=(UNITS, TS_LEN), scale=3)
    locations = rng.integers(0, TS_LEN - MOTIF_LEN, size=len(noise))
    locations[INJECT:] = -1
    rng.shuffle(locations)
    for i, loc in enumerate(locations):
        if loc == -1:
            continue
        else:
            # Inject motif
            noisy_motif = MOTIF + rng.normal(size=MOTIF_LEN, scale=noise_level)
            noise[i][loc : loc + MOTIF_LEN] = noisy_motif

    return zscore(np.cumsum(noise, axis=1), axis=1), locations


def ostinato(data, m, precomputed=None):
    if precomputed is None:
        radius, idx, subidx = stumpy.ostinato(data, m)
        print(idx, subidx)
    else:
        idx, subidx = precomputed
    return data[idx][subidx : subidx + m]


if __name__ == "__main__":
    main()
