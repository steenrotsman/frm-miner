from collections import namedtuple
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from plot import HEIGHT, WIDTH, plot_motifs

from frm import Miner

# Simulation settings
UNITS = 10000
TS_LEN = 10000
RNG = np.random.default_rng(1234)
NOISE = 2
Setting = namedtuple("Setting", ["name", "support", "motif"])

LENGTHS = [200, 250, 300, 350]
SETTINGS = [
    Setting("up", 9000, np.linspace(-3, 3, LENGTHS[0]) ** 2),
    Setting("peak", 8500, np.linspace(4, -4, LENGTHS[1])),
    Setting("down", 8000, -(np.linspace(-3, 3, LENGTHS[2]) ** 2)),
    Setting("dip", 7500, np.linspace(-4, 4, LENGTHS[3])),
]


# Starting indices for motifs in each time series
LOCATIONS = [
    RNG.integers(0, 1500, size=UNITS),
    RNG.integers(2500, 4000, size=UNITS),
    RNG.integers(5000, 6500, size=UNITS),
    RNG.integers(7500, 9000, size=UNITS),
]

# Parameters
MINSUP = 0.3
SEGLEN = 30
ALPHA = 4
OMAX = 0.6
DIFF = 1
K = 4


def main():
    ts, ground_truth = get_data()

    fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(WIDTH, HEIGHT))
    # Select rows with and without motifs
    for row, indices in zip(ts, ground_truth):
        ts_index = np.where(indices == -1)[0][0]
        ax.plot(ts[ts_index])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(join("figs", "3 bigdata.png"))

    motifs = get_motifs(ts)
    for motif in motifs:
        print(motif.length, len(motif.best_matches))
    fig, axs = plt.subplots(ncols=K, figsize=(WIDTH, HEIGHT), sharey="all")
    plot_motifs(axs, ts, motifs, fn="3 big final")


def get_data():
    # Gaussian noise
    ts = RNG.normal(0, NOISE, size=(UNITS, TS_LEN))
    # Only keep starting indices for motif-specific support number of time series
    ground_truth = []
    for locations, length, setting in zip(LOCATIONS, LENGTHS, SETTINGS):
        locations[setting.support :] = -1
        RNG.shuffle(locations)
        ground_truth.append(locations)
        for i, loc in enumerate(locations):
            if loc == -1:
                continue
            noisy_motif = setting.motif + RNG.normal(size=length)
            ts[i][loc : loc + length] = noisy_motif
    return np.cumsum(ts, axis=1), ground_truth


def get_motifs(ts):
    mm = Miner(MINSUP, SEGLEN, ALPHA, k=K, diff=DIFF, omax=OMAX)
    motifs = mm.mine(ts)
    return motifs


if __name__ == "__main__":
    main()
