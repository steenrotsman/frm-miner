from os.path import join
from statistics import fmean

import matplotlib.pyplot as plt

import plot  # noqa

FILE = "e9_params.csv"

MINSUP = [0.1, 0.3, 0.5, 0.7, 0.9]
SEGLEN = [10, 20, 30, 40, 50]
ALPHA = [2, 3, 4, 5, 6, 7, 8, 9, 10]
OMAX = [0.5, 0.6, 0.7, 0.8, 0.9]
PARAMS = ["minsup", "seglen", "α", "omax"]
VALUES = [MINSUP, SEGLEN, ALPHA, OMAX]


minsup_runtime = [[] for _ in MINSUP]
seglen_runtime = [[] for _ in SEGLEN]
alpha_runtime = [[] for _ in ALPHA]
omax_runtime = [[] for _ in OMAX]
minsup_n_patterns = [[] for _ in MINSUP]
seglen_n_patterns = [[] for _ in SEGLEN]
alpha_n_patterns = [[] for _ in ALPHA]
omax_n_patterns = [[] for _ in OMAX]

with open(FILE) as fp:
    for row in fp.readlines():
        _, m, s, a, o, r, n = tuple(row[:-1].split(","))
        m = float(m)
        s = int(s)
        a = int(a)
        o = float(o)
        r = float(r)
        n = int(n)

        if m not in MINSUP or s not in SEGLEN or a not in ALPHA or o not in OMAX:
            continue

        minsup_runtime[MINSUP.index(m)].append(r)
        seglen_runtime[SEGLEN.index(s)].append(r)
        alpha_runtime[ALPHA.index(a)].append(r)
        omax_runtime[OMAX.index(o)].append(r)
        minsup_n_patterns[MINSUP.index(m)].append(n)
        seglen_n_patterns[SEGLEN.index(s)].append(n)
        alpha_n_patterns[ALPHA.index(a)].append(n)
        omax_n_patterns[OMAX.index(o)].append(n)

results = [
    ([fmean(x) for x in minsup_runtime], [fmean(x) for x in minsup_n_patterns]),
    ([fmean(x) for x in seglen_runtime], [fmean(x) for x in seglen_n_patterns]),
    ([fmean(x) for x in alpha_runtime], [fmean(x) for x in alpha_n_patterns]),
    ([fmean(x) for x in omax_runtime], [fmean(x) for x in omax_n_patterns]),
]

fig, axs = plt.subplots(ncols=4, sharey="all")
for (runtime, n_patterns), ax, xlabel, xticks in zip(results, axs, PARAMS, VALUES):
    twin = ax.twinx()
    time = ax.plot(xticks, runtime, "k", label="Time")
    patterns = twin.plot(xticks, n_patterns, "k--", label="Patterns")
    ax.set(xlabel=xlabel, xticks=xticks)
    twin.set_ylim(0, 100)
    twin.set_yticks([0, 50, 100], labels="")

axs[0].set(ylabel="Seconds", yticks=[0, 20, 40])
twin.set(ylabel="n Patterns")
twin.set_yticks([0, 50, 100], labels=[0, 50, 100])
fig.legend(handles=[time[0], patterns[0]], loc="outside lower center", ncols=2)

plt.savefig(join("figs", "Fig16.pdf"))
plt.close()
