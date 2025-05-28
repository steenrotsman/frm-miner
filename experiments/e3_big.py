from collections import namedtuple
from multiprocessing import Pool
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from plot import WIDTH, plot_motifs
from scipy.stats import zscore

from frm import Miner

# Simulation settings
UNITS = 10000
TS_LEN = 10000
RNG = np.random.default_rng(1234)

Setting = namedtuple("Setting", ["name", "length", "support", "noise"])

SETTINGS = [
    Setting("peak", 300, 9000, 1),
    Setting("trough", 350, 8000, 1),
    Setting("up", 400, 7000, 1),
    Setting("down", 450, 6000, 1),
    Setting("constant", 500, 5000, 1),
]
MOTIFS = [
    zscore(np.arange(0, SETTINGS[0].length)),
    zscore(np.arange(SETTINGS[1].length, 0, -1)),
    np.ones(SETTINGS[2].length) / 5,
    -np.ones(SETTINGS[3].length) / 5,
    np.zeros(SETTINGS[4].length),
]


# Starting indices for motifs in each time series
LOCATIONS = [
    RNG.integers(0, 1000, size=UNITS),
    RNG.integers(2000, 3000, size=UNITS),
    RNG.integers(4000, 5000, size=UNITS),
    RNG.integers(6000, 7000, size=UNITS),
    RNG.integers(8000, 9000, size=UNITS),
]

# Parameters
MINSUP = 0.3
SEGLEN = [30, 50, 70, 75, 100]
ALPHA = 4
DIFF = 1
K = 5


def main():
    ts = get_data()

    with Pool(len(SEGLEN)) as p:
        motifss = p.starmap(get_motifs, [(ts, seglen) for seglen in SEGLEN])

    fig, axss = plt.subplots(ncols=K, nrows=len(SEGLEN), figsize=(WIDTH, WIDTH))
    for motifs, axs in zip(motifss, axss):
        plot_motifs(axs, ts, motifs)
    plt.savefig(join("figs", "3 big.png"))
    plt.savefig(join("figs", "3 big.eps"))


def get_data():
    # Gaussian noise
    noise = RNG.normal(0, 5, size=(UNITS, TS_LEN))

    # Only keep starting indices for motif-specific support number of time series
    for locations, setting, motif in zip(LOCATIONS, SETTINGS, MOTIFS):
        locations[setting.support :] = -1
        RNG.shuffle(locations)
        for i, loc in enumerate(locations):
            if loc == -1:
                continue
            noisy_motif = motif + RNG.normal(size=setting.length, scale=setting.noise)
            noise[i][loc : loc + setting.length] = noisy_motif
    return zscore(np.cumsum(noise, axis=1), axis=1)


def get_motifs(ts, seglen):
    mm = Miner(MINSUP, seglen, ALPHA, k=K, diff=DIFF)
    motifs = mm.mine(ts)
    return motifs


if __name__ == "__main__":
    main()
