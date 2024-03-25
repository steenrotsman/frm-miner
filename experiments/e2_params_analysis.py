from os.path import join
from statistics import fmean

import matplotlib.pyplot as plt
import plot  # noqa

FILE = 'e2_params.csv'

MINSUP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEGLEN = [10, 15, 20, 25, 30, 35, 40, 45, 50]
ALPHABET = [2, 3, 4, 5, 6, 7, 8, 9, 10]
OMAX = [0.5, 0.6, 0.7, 0.8, 0.9]
PARAMS = ['minsup', 'seglen', 'alphabet', 'omax']
VALUES = [MINSUP, SEGLEN, ALPHABET]
BAR_WIDTH = 0.45


minsup_runtime = [[] for _ in MINSUP]
seglen_runtime = [[] for _ in SEGLEN]
alphabet_runtime = [[] for _ in ALPHABET]
omax_runtime = [[] for _ in OMAX]
minsup_n_patterns = [[] for _ in MINSUP]
seglen_n_patterns = [[] for _ in SEGLEN]
alphabet_n_patterns = [[] for _ in ALPHABET]
omax_n_patterns = [[] for _ in OMAX]

with open(FILE) as fp:
    for row in fp.readlines():
        _, m, s, a, o, r, n = tuple(row[:-1].split(','))
        m = float(m)
        s = int(s)
        a = int(a)
        o = int(o)
        r = float(r)
        n = int(n)

        if m not in MINSUP or s not in SEGLEN or a not in ALPHABET or o not in OMAX:
            continue

        minsup_runtime[MINSUP.index(m)].append(r)
        seglen_runtime[SEGLEN.index(s)].append(r)
        alphabet_runtime[ALPHABET.index(a)].append(r)
        omax_runtime[OMAX.index(o)].append(r)
        minsup_n_patterns[MINSUP.index(m)].append(n)
        seglen_n_patterns[SEGLEN.index(s)].append(n)
        alphabet_n_patterns[ALPHABET.index(a)].append(n)
        omax_n_patterns[OMAX.index(o)].append(n)

results = [
    ([fmean(x) for x in minsup_runtime], [fmean(x) for x in minsup_n_patterns]),
    ([fmean(x) for x in seglen_runtime], [fmean(x) for x in seglen_n_patterns]),
    ([fmean(x) for x in alphabet_runtime], [fmean(x) for x in alphabet_n_patterns]),
    ([fmean(x) for x in omax_runtime], [fmean(x) for x in omax_n_patterns]),
]

fig, axs = plt.subplots(ncols=4, sharey='all')
for (runtime, n_patterns), ax, xlabel, xticks in zip(results, axs, PARAMS, VALUES):
    twin = ax.twinx()
    time = ax.plot(xticks, runtime, 'k', label='Time')
    patterns = twin.plot(xticks, n_patterns, 'k--', label='Patterns')
    ax.set(xlabel=xlabel, xticks=xticks)
    ax.tick_params(axis='x', labelrotation=-90)
    twin.set(yticks=[], ylim=[0, 400])

axs[0].set(ylabel='Seconds', yticks=[0, 20, 40])
twin.set(ylabel='n Patterns', yticks=[0, 200, 400])
twin.legend(handles=[time[0], patterns[0]])

plt.savefig(join('figs', '2 params.eps'))
plt.savefig(join('figs', '2 params.png'))
plt.close()
