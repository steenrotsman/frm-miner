from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import stumpy
from plot import plot_motifs, remove_spines
from scipy.stats import zscore

from frm import Miner

# Simulation settings
UNITS = 100
TS_LEN = 10000
MOTIF_LEN = 500
NOISE_LEVELS = [0.2, 0.4, 0.6, 0.8]
INJECT = 75

# Parameters
MINSUP = 0.3
SEGLEN = 25
ALPHA = 4
K = 3

RNG = np.random.default_rng(0)


def main():
    data = get_data(UNITS, TS_LEN, MOTIF_LEN, INJECT, NOISE_LEVELS)
    fix, axs = plt.subplots(nrows=3, sharex='all')
    for row, ax in zip(data[0.6], axs):
        ax.plot(row, 'k')
        remove_spines(ax)
        ax.set_xticks([0, len(row)])
    plt.savefig(join('figs', '3 data.eps'))
    plt.savefig(join('figs', '3 data.png'))

    # FRM-Miner 1.0 and 2.0
    for p in (1, 2):
        fig, axss = plt.subplots(K, len(NOISE_LEVELS), sharex='all', sharey='all')
        for noise_level, axs in zip(NOISE_LEVELS, axss.T):
            rows = data[noise_level]

            mm = Miner(MINSUP, SEGLEN, ALPHA, k=K, p=p)
            motifs = mm.mine(rows)
            plot_motifs(axs, zscore(rows, axis=1), motifs, MOTIF_LEN)
            axs[0].set_yticks([-2.5, 2.5])
            axs[2].set_xlabel(noise_level)

        plt.savefig(join('figs', f'3 accuracy {p}.eps'))
        plt.savefig(join('figs', f'3 accuracy {p}.png'))
        plt.close()

    # Ostinato
    consensus_motifs = [ostinato(data[n], MOTIF_LEN) for n in NOISE_LEVELS]
    plot_ostinato(consensus_motifs, INJECT)


def get_data(units, ts_len, motif_len, inject, noise_levels):
    data = {}

    for noise_level in noise_levels:
        # Generate noise and motif
        noise = RNG.normal(size=(units, ts_len - motif_len))
        steps = np.linspace(0, np.pi, motif_len)
        motif = np.sin(3 * steps) + np.sin(4 * steps)

        # Randomly select locations where to inject noisy motif
        locations = RNG.integers(0, noise.shape[1], size=len(noise))
        locations[inject:] = -1
        RNG.shuffle(locations)

        # Embed motifs in noise
        rows = []
        for row, loc in zip(noise, locations):
            if loc == -1:
                # Add extra random noise to row
                row = np.hstack((row, RNG.normal(size=motif_len)))
            else:
                # Add noisy motif to row
                noisy_motif = motif + RNG.normal(scale=noise_level, size=motif_len)
                row = np.hstack((row[:loc], noisy_motif, row[loc:]))
            rows.append(row)

        # Add data to simulation data
        data[noise_level] = np.array(rows)

    return data


def ostinato(data, m, i=[0]):
    # Pre-computed points, comment out until first return to calculate consensus motifs
    points = [(36, 8799), (91, 7012), (4, 3063), (81, 4829)]
    idx, subidx = points[i[0]]
    consensus_motif = data[idx][subidx : subidx + m]

    # Terrible hack
    i[0] = i[0] + 1
    return consensus_motif

    # Warning: takes long
    radius, idx, subidx = stumpy.ostinato(data, m)
    print(idx, subidx)
    return data[idx][subidx : subidx + m]


def plot_ostinato(consensus_motifs, inject):
    fig, axs = plt.subplots(ncols=len(consensus_motifs), sharex='all', sharey='all')

    for motif, ax, noise_level in zip(consensus_motifs, axs, NOISE_LEVELS):
        ax.plot(motif, 'k', lw=0.5)
        ax.set(ylim=(-3, 3), xticks=[0, len(motif)], yticks=[-2.5, 2.5])
        ax.set_xlabel(noise_level)
        remove_spines(ax, remove_y=False)

    plt.savefig(join('figs', '3 accuracy ostinato.eps'))
    plt.savefig(join('figs', '3 accuracy ostinato.png'))


if __name__ == '__main__':
    main()
