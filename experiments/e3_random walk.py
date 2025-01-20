from os.path import join
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from plot import remove_spines
from scipy.stats import zscore

from frm import Miner

# Simulation settings
UNITS = 10
TS_LEN = 10000
MOTIF_LEN = 512
INJECT = 100

# Parameters
MINSUP = 0.6
SEGLEN = 32
ALPHA = 4
K = 1

RNG = np.random.default_rng(3141657)


def main():
    motif = np.sin(np.arange(0.1, 9.6, 0.01856))
    data = []

    for i in range(UNITS):
        ts = zscore(np.cumsum(RNG.normal(size=TS_LEN - MOTIF_LEN)))
        ts = zscore(np.concatenate((ts, (motif * (RNG.random() + 0.1)) + ts[-1])))
        ts = zscore(ts + RNG.random(len(ts)) / 20)
        data.append(ts)
    data = np.array(data)
    plt.plot(data.T)
    plt.savefig(join('figs', '3 walks.png'))
    data = np.diff(data, axis=1)
    plt.savefig(join('figs', '3 diff.png'))

    start = perf_counter()
    mm = Miner(MINSUP, SEGLEN, ALPHA)
    motifs = mm.mine(data)
    end = perf_counter()
    print(end - start)

    fig, ax = plt.subplots()
    ax.plot(motifs[0].representative, 'b', lw=1)
    ax.set(xticks=[0, len(motifs[0].representative)])
    remove_spines(ax, remove_y=False)
    plt.savefig(join('figs', '3 walk.png'))
    plt.close()


if __name__ == '__main__':
    main()
