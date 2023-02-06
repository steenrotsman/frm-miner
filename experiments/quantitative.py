"""
This module provides a quantitative experiment that injects different motifs into random data to check retrieval.

A random noise time series database is generated, into which constructed motifs are injected.
The proposed algorithm is then run and the motifs it finds are plotted.
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt
import stumpy

from motifminer.miner import Miner
from motifminer.preprocessing import breakpoints

# Simulation settings
UNITS = 100
TS_LEN = 10000
MOTIF_LEN = 500
NOISE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Parameters
MIN_SUP = 0.5
SEGMENT = 25
ALPHABET = 5
MIN_LEN = 3
MAX_OVERLAP = 0.8
LOCAL = False
K = 5

np.random.seed(42)


def main():
    noise = np.random.normal(size=(UNITS, TS_LEN - MOTIF_LEN))
    motif = get_motif(MOTIF_LEN)
    plot_motif(motif, NOISE_LEVELS)
    plot_example(noise[:5], motif, 0.5)

    motifs = []
    for noise_level in NOISE_LEVELS:
        # Put noisy motif into data
        data = get_data(noise, motif, noise_level)

        # plot_ostinato(data, MOTIF_LEN)

        # Mine motifs of variable length
        mm = Miner(data, MIN_SUP, SEGMENT, ALPHABET, MIN_LEN, MAX_OVERLAP, LOCAL, K)
        motifs.append(mm.mine_motifs())

    plot_motifs(motifs, NOISE_LEVELS, MOTIF_LEN)


def get_motif(length):
    steps = np.linspace(0, np.pi, length)
    waves = np.sin(2 * steps) + np.sin(3 * steps)

    return waves


def get_data(noise, motif, noise_level):
    data = []
    motif_len = len(motif)

    for i, row in enumerate(noise):
        # Add random noise to motif
        noisy_motif = motif + np.random.normal(scale=noise_level, size=motif_len)

        # Inject motif into row at different positions
        loc = np.random.randint(0, noise.shape[1])
        row = np.hstack((row[:loc], noisy_motif, row[loc:]))

        # Add row to simulation data
        data.append(row)

    return data


def plot_motif(motif, noise_levels):
    fig, axs = plt.subplots(ncols=len(noise_levels), sharex='all', sharey='all')
    fig.set_dpi(300)

    for noise_level, ax in zip(noise_levels, axs):
        for j in range(1):
            ax.plot(motif + np.random.normal(scale=noise_level, size=len(motif)), 'k', linewidth=0.3)
            remove_spines(ax)

    plt.show()


def plot_example(noise, motif, noise_level):
    data = get_data(noise, motif, noise_level)
    fig, axs = plt.subplots(nrows=noise.shape[0], sharex='all', sharey='all')
    fig.set_dpi(300)

    for row, ax in zip(data, axs):
        ax.plot(row, 'k')
        remove_spines(ax)

    plt.show()


def plot_motifs(motifs, noise_levels, max_length):
    fig, axs = plt.subplots(nrows=len(noise_levels), ncols=len(motifs[0]), sharey='all', sharex='all')
    fig.set_dpi(300)

    flat_motifs = itertools.chain.from_iterable(motifs)
    flat_axs = itertools.chain.from_iterable(axs)
    for motif, ax in zip(flat_motifs, flat_axs):
        ax.plot(motif.matches, 'k', lw=0.1)
        ax.plot(motif.representative, 'b', lw=1)
        breaks = list(breakpoints[ALPHABET].keys())[:-1]
        ax.hlines(breaks, 0, max_length, 'k', lw=0.3)
        remove_spines(ax)
        ax.set_xticks([0, max_length])

    fig.tight_layout()
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


def remove_spines(ax):
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


if __name__ == '__main__':
    main()
