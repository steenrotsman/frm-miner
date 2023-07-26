from statistics import fmean
from math import log10 as log

import matplotlib.pyplot as plt
from matplotlib import rcParams

from plot import remove_spines, COLORS

FILE = 'e2_params.csv'

# Parameter ranges for FRM-Miner
MINSUP = [0.1, 0.3, 0.5, 0.7]
SEGLEN = [5, 10, 20, 50]
ALPHABET = [3, 5, 7, 9]
BAR_WIDTH = 0.45


m_r = [[] for _ in MINSUP]
s_r = [[] for _ in SEGLEN]
a_r = [[] for _ in ALPHABET]
m_n = [[] for _ in MINSUP]
s_n = [[] for _ in SEGLEN]
a_n = [[] for _ in ALPHABET]

with open(FILE) as fp:
    for row in fp.readlines():
        m, s, a, r, n = tuple(row[:-1].split(','))
        m = float(m)
        s = int(s)
        a = int(a)
        r = float(r)
        n = int(n)

        if m not in MINSUP or s not in SEGLEN or a not in ALPHABET:
            continue

        m_r[MINSUP.index(m)].append(r)
        s_r[SEGLEN.index(s)].append(r)
        a_r[ALPHABET.index(a)].append(r)
        m_n[MINSUP.index(m)].append(n)
        s_n[SEGLEN.index(s)].append(n)
        a_n[ALPHABET.index(a)].append(n)

m_r = [int(round(fmean(x), 0)) for x in m_r]
s_r = [int(round(fmean(x), 0)) for x in s_r]
a_r = [int(round(fmean(x), 0)) for x in a_r]
m_n = [int(round(fmean(x), 0)) for x in m_n]
s_n = [int(round(fmean(x), 0)) for x in s_n]
a_n = [int(round(fmean(x), 0)) for x in a_n]

rcParams['font.size'] = 3
fig, axs = plt.subplots(ncols=3, layout='constrained', sharey='all')
for results, ax, xlabel, xticks in zip([(m_r, m_n), (s_r, s_n), (a_r, a_n)], axs, ['minsup', 'seglen', 'alphabet'], [MINSUP, SEGLEN, ALPHABET]):
    bar_positions1 = [x - BAR_WIDTH/2 for x in range(len(xticks))]
    bar_positions2 = [x + BAR_WIDTH/2 for x in range(len(xticks))]

    ax.bar(bar_positions1, [log(x) for x in results[0]], BAR_WIDTH, color=COLORS[0], label='Runtime (seconds)')
    ax.bar(bar_positions2, [log(x) for x in results[1]], BAR_WIDTH, color=COLORS[1], label='n motifs')

    for i in range(len(xticks)):
        ax.text(bar_positions1[i], log(results[0][i]), str(results[0][i]), ha='center', va='bottom')
        ax.text(bar_positions2[i], log(results[1][i]), str(results[1][i]), ha='center', va='bottom')

    # Set the x-axis ticks
    ax.set_xticks(list(range(len(xticks))))
    ax.set_xticklabels(xticks)

    remove_spines(ax)

plt.savefig(f'figs/2 params.eps')
plt.close()
