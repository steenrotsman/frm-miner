from os.path import join
from time import perf_counter
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
from plot import remove_spines
from scipy.stats import zscore

from frm import Miner

with catch_warnings():
    simplefilter("ignore")
    from mass_ts import mass2 as mass

# Simulation settings
UNITS = 10
TS_LEN = 10000
MOTIF_LEN = 512
INJECT = 100

# Parameters
MINSUP = 0.3
SEGLEN = 25
ALPHA = 4
K = 1

RNG = np.random.default_rng(3141657)


def main():
    # Generate data
    motif = np.sin(np.arange(0.1, 9.6, 0.01856))
    data = []

    for i in range(UNITS):
        ts = zscore(np.cumsum(RNG.normal(size=TS_LEN - MOTIF_LEN)))
        ts = zscore(np.concatenate((ts, (motif * (RNG.random() + 0.1)) + ts[-1])))
        ts = zscore(ts + RNG.random(len(ts)) / 20)
        data.append(ts)
    data = np.array(data)

    # Plot data
    plt.plot(data.T, lw=0.3)
    plt.savefig(join('figs', '3 walks.png'))
    plt.close()
    diff = np.diff(data, axis=1)
    plt.plot(diff.T, lw=0.3)
    plt.savefig(join('figs', '3 diff.png'))
    plt.close()

    # Discover motifs
    start = perf_counter()
    mm = Miner(MINSUP, SEGLEN, ALPHA)
    motifs = mm.mine(diff)
    end = perf_counter()
    print(end - start)

    # Plot motifs
    fig, ax = plt.subplots()
    motif = motifs[0]
    for ts, idx in motif.best_matches.items():
        ax.plot(zscore(data[ts][idx : idx + motif.length]), lw=0.3)
    ax.set(xticks=[0, motif.length])
    remove_spines(ax, remove_y=False)
    plt.savefig(join('figs', '3 walk.png'))

    # Plot additional occurrences
    representative = np.mean(
        [
            zscore(data[ts][idx : idx + motif.length])
            for ts, idx in motif.best_matches.items()
        ],
        axis=0,
    )
    for i, series in enumerate(data):
        if i not in motif.best_matches:
            with catch_warnings():
                simplefilter("ignore")
                idx = np.argmin(mass(data[i], representative))
            ax.plot(zscore(data[i][idx : idx + motif.length]), lw=0.3)

    plt.savefig(join('figs', '3 additional.png'))
    plt.close()


if __name__ == '__main__':
    main()
