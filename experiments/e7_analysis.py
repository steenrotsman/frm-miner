from collections import defaultdict
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np
from e7_ucr import ALPHA, SEGLEN
from plot import remove_spines

FILE = "e7_ucr.csv"

experiments = defaultdict(list)
with open(FILE) as fp:
    for row in fp:
        *experiment, sup, m, s, a, f = row.strip().split(",")
        experiments[tuple(experiment)].append((float(m), int(s), int(a), eval(f)))

# Count n successes for each experiment
success_counts = []
for experiment, results in experiments.items():
    count = sum(1 for result in results if result[-1])
    success_counts.append(count)
x_values = np.arange(1, 41)
y_values = []
for x in x_values:
    fraction = sum(1 for count in success_counts if count >= x) / len(success_counts)
    y_values.append(fraction)

print(round(sum(success_counts) / (len(success_counts) * 40), 3) * 100)
print(round(y_values[0], 3) * 100)
print(round(y_values[-1], 3) * 100)

fix, ax = plt.subplots()
ax.plot(x_values, y_values, marker=".", linestyle="-", color="k")
ax.set_xlabel("Parameter combinations with retrieval")
ax.set_ylabel("Success rate")
ax.set_yticks([0.8, 0.9, 1.0])
ax.set_xticks(range(5, 45, 5))
remove_spines(ax, False)
plt.savefig("figs/7 success.eps")
plt.savefig("figs/7 success.png")


# Success rates for seglen and alpha
seglen_success = defaultdict(lambda: [0, 0])
alpha_success = defaultdict(lambda: [0, 0])
for settings in experiments.values():
    for setting in settings:
        minsup, seglen, alpha, success = setting
        seglen_success[seglen][1] += 1
        alpha_success[alpha][1] += 1
        if success:
            seglen_success[seglen][0] += 1
            alpha_success[alpha][0] += 1

fix, axs = plt.subplots(ncols=2, sharey="all")
x = zip([seglen_success, alpha_success], [SEGLEN, ALPHA], ["seglen", "α"], axs)
for param, values, label, ax in x:
    success_rates = [param[v][0] / param[v][1] for v in param]
    ax.plot(values, success_rates, marker=".", linestyle="-", color="k")
    ax.set_xlabel(label)
    remove_spines(ax, False)
axs[0].set_ylabel("Success Rate")
seglen_success = [seglen_success[v][0] / seglen_success[v][1] for v in seglen_success]
ticks = [min(seglen_success), fmean(success_rates), max(seglen_success)]
axs[0].set_yticks([round(tick, 3) for tick in ticks])
axs[0].set_xticks(range(1, 11))
plt.savefig("figs/7 params.eps")
plt.savefig("figs/7 params.png")
