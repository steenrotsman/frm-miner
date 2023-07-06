from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

from plot import remove_spines

FILE = '2_params.csv'

# Parameter ranges for FRM-Miner
MINSUP = [0.1, 0.3, 0.5, 0.7, 0.9]
SEGLEN = [5, 10, 20, 50, 100]
ALPHABET = [2, 3, 4, 5, 6, 7, 8, 9, 10]

m_r = defaultdict(list)
s_r = defaultdict(list)
a_r = defaultdict(list)
m_n = defaultdict(list)
s_n = defaultdict(list)
a_n = defaultdict(list)

with open(FILE) as fp:
    for row in fp.readlines():
        m, s, a, r, n = tuple(row[:-1].split(','))
        m = float(m)
        s = int(s)
        a = int(a)

        m_r[(s, a)].append(r)
        s_r[(m, a)].append(r)
        a_r[(m, s)].append(r)
        m_n[(s, a)].append(n)
        s_n[(m, a)].append(n)
        a_n[(m, s)].append(n)

for results, name, axis in zip([(m_r, m_n), (s_r, s_n), (a_r, a_n)], ['minsup', 'seglen', 'alphabet'], [MINSUP, SEGLEN, ALPHABET]):
    fig, axs = plt.subplots(nrows=2, sharex='all', layout='constrained')

    for ax, res, label in zip(axs, results, ['runtime', 'n motifs']):
        series = []
        for r in res.values():
            serie = zscore([float(x) for x in r])
            series.append(serie)
            ax.plot(serie, lw=0.1, color='k')
        ax.plot(np.mean(series, axis=0), lw=0.5, color='b')
        ax.set_xticks(list(range(len(axis))), labels=axis)
        ax.set_yticks([])
        ax.set(ylabel=label)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    ax.set(xlabel=name)
    plt.savefig(f'figs/2 params {name}')
    plt.close()