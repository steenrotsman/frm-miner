"""
This module provides a quantitative experiment that injects different motifs into random data to check retrieval.

A random noise time series database is generated, into which constructed motifs are injected.
The proposed algorithm is then run and the motifs it finds are plotted.
"""
from itertools import chain
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import stumpy

from frm._frm_py.miner import Miner
from plot import plot_motifs, remove_spines

# Simulation settings
UNITS = 100
TS_LEN = 10000
MOTIF_LEN = 500
NOISE_LEVELS = [0.1, 0.5, 0.9]
INJECT = [100, 75]

# Parameters
MINSUP = 0.3
SEGLEN = 30
ALPHABET = 4
K = 3

np.random.seed(0)


def main():
    noise = np.random.normal(size=(UNITS, TS_LEN - MOTIF_LEN))
    motif = get_motif(MOTIF_LEN)

    for inject in INJECT:
        motifs = []
        consensus_motifs = []
        for noise_level in NOISE_LEVELS:
            # Put noisy motif into data
            data, locations = get_data(noise, motif, noise_level, inject)

            # Mine motifs of variable length
            mm = Miner(MINSUP, SEGLEN, ALPHABET, k=K)
            top_motifs = mm.mine_motifs(data)
            motifs.append(top_motifs)

            # Find consensus motif
            consensus_motifs.append(ostinato(data, MOTIF_LEN))

        fig, axs = plt.subplots(nrows=len(motifs[0]), layout='compressed', ncols=len(NOISE_LEVELS), sharey='all', sharex='all')
        plot_motifs(fig, chain.from_iterable(axs.T), chain.from_iterable(motifs), ALPHABET, MOTIF_LEN, fn=f'3 accuracy {inject}')
        plot_ostinato(consensus_motifs, inject)


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


def ostinato(data, m, i=[0]):
    # Pre-computed points, comment out until line 129 to calculate consensus motifs again
    points = [
        (58, 3998), (59, 4032), (40, 8878),
        (42, 2178), (74, 8804), (92, 6033),
    ]

    idx, subidx = points[i[0]]
    consensus_motif = data[idx][subidx : subidx+m]

    # Terrible hack
    i[0] = i[0] + 1
    return consensus_motif

    # Warning: takes long
    radius, idx, subidx = stumpy.ostinato(data, m)
    print(idx, subidx)
    return data[idx][subidx : subidx+m]


def plot_ostinato(consensus_motifs, inject):
    fig, axs = plt.subplots(ncols=len(consensus_motifs), layout='compressed')

    for motif, ax in zip(consensus_motifs, axs):
        ax.plot(motif, 'k', lw=0.5)
        ax.set(ylim=(-3, 3), xticks=[0, len(motif)], yticks=[])
        remove_spines(ax)

    plt.savefig(join('figs', f'3 accuracy {inject} ostinato'))


if __name__ == '__main__':
    main()
