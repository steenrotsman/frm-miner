"""
This module provides a quantitative experiment that injects different motifs into random data to check retrieval.

A random noise time series database is generated, into which constructed motifs are injected.
The proposed algorithm is then run and the motifs it finds are plotted.
"""
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import stumpy

from motifminer.miner import Miner
from plot import plot_motifs, remove_spines

# Simulation settings
UNITS = 100
TS_LEN = 10000
MOTIF_LEN = 500
NOISE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
INJECT = 75

# Parameters
MIN_SUP = 0.5
SEGMENT = 20
ALPHABET = 4
MIN_LEN = 3
MAX_OVERLAP = 0.8
LOCAL = True
K = 3

np.random.seed(42)


def main():
    noise = np.random.normal(size=(UNITS, TS_LEN - MOTIF_LEN))
    motif = get_motif(MOTIF_LEN)
    plot_motif(motif, NOISE_LEVELS)
    plot_example(noise[:5], motif, 0.5)

    motifs = []
    overlaps = []
    for noise_level in NOISE_LEVELS:
        # Put noisy motif into data
        data, locations = get_data(noise, motif, noise_level, INJECT)

        # Mine motifs of variable length
        mm = Miner(data, MIN_SUP, SEGMENT, ALPHABET, MIN_LEN, MAX_OVERLAP, LOCAL, K)
        top_motifs = mm.mine_motifs()
        motifs.append(top_motifs)

        # Find if occurrences of top_motifs correspond to actual locations
        overlap = get_overlap(top_motifs, locations)
        overlaps.append(overlap)

    fig, axs = plt.subplots(nrows=len(motifs[0]), figsize=(15, 3), ncols=len(NOISE_LEVELS), sharey='all', sharex='all')
    plot_motifs(fig, chain.from_iterable(axs.T), chain.from_iterable(motifs), ALPHABET, MOTIF_LEN)
    print(np.array(overlaps).T)


def get_motif(length):
    steps = np.linspace(0, np.pi, length)
    waves = np.sin(3 * steps) + np.sin(4 * steps)

    return waves


def get_data(noise, motif, noise_level, inject):
    data = []
    motif_len = len(motif)

    # Randomly select locations where to inject noisy motif
    locations = np.random.randint(0, noise.shape[1], size=len(noise))
    locations[inject:] = -1
    np.random.shuffle(locations)

    for row, loc in zip(noise, locations):
        if loc == -1:
            # Add extra random noise to row
            row = np.hstack((row, np.random.normal(size=len(motif))))
        else:
            # Add noisy motif to row
            noisy_motif = motif + np.random.normal(scale=noise_level, size=motif_len)
            row = np.hstack((row[:loc], noisy_motif, row[loc:]))

        # Add row to simulation data
        data.append(row)

    return data, locations


def get_overlap(top_motifs, locations):
    overlap = []
    for motif in top_motifs:
        success = 0
        for i, (start, end) in motif.match_indexes.items():
            true_start = locations[i]
            true_end = locations[i] + MOTIF_LEN

            # Check if match doesn't completely fall outside of range
            if true_start < 0 or start > true_end or end < true_start:
                continue

            if start < true_start:
                success += end - true_start > true_start - start
            elif end > true_end:
                success += end - true_end < true_end - start
            else:
                success += True
        overlap.append(success / len(motif.match_indexes))

    return overlap


def plot_motif(motif, noise_levels):
    fig, axs = plt.subplots(ncols=len(noise_levels), figsize=(15, 3), sharex='all', sharey='all')
    fig.set_dpi(300)
    fig.tight_layout()

    for noise_level, ax in zip(noise_levels, axs):
        for j in range(1):
            ax.plot(motif + np.random.normal(scale=noise_level, size=len(motif)), 'k', linewidth=0.3)
            remove_spines(ax)
            ax.set_xticks([0, len(motif)])

    plt.show()


def plot_example(noise, motif, noise_level):
    data, _ = get_data(noise, motif, noise_level, len(noise))
    fig, axs = plt.subplots(nrows=noise.shape[0], figsize=(15, 3), sharex='all', sharey='all')
    fig.set_dpi(300)
    fig.tight_layout()

    for row, ax in zip(data, axs):
        ax.plot(row, 'k')
        remove_spines(ax)
        ax.set_xticks([0, len(row)])
        ax.margins(0.01)

    plt.show()


def plot_ostinato(data, m):
    radius, idx, subidx = stumpy.ostinato(data, m)
    consensus_motif = data[idx][subidx : subidx+m]

    # Plot consensus motif and best match in every ts
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    nn_idxs = []
    for i, ts in enumerate(data):
        if i == idx:
            c, lw = 'b', 1
        else:
            c, lw = 'k', 0.01
        nn_idxs.append(np.argmin(stumpy.core.mass(consensus_motif, ts)))
        ax.plot(stumpy.core.z_norm(ts[nn_idxs[i] : nn_idxs[i]+m]), c, lw=lw)
    plt.show()


if __name__ == '__main__':
    main()
