from collections import defaultdict
from os.path import join

import matplotlib.pyplot as plt

from .plot import WIDTH

FILE = 'e1_runtime.csv'
SIZE = 0.3
METHOD_1 = 'benchmark_cpp_miner'
METHOD_2 = 'benchmark_stumpy'

method_1_runtimes = {}
method_2_runtimes = {}
runtimes = defaultdict(float)
with open(FILE) as fp:
    for line in fp.readlines():
        row = line[:-1].split(',')
        method, dataset, runtime = row
        runtime = float(runtime)
        runtimes[method] += runtime

        if method == METHOD_1:
            method_1_runtimes[dataset] = float(runtime)
        elif method == METHOD_2:
            method_2_runtimes[dataset] = float(runtime)

# Print total run times
for v, method in zip(runtimes.values(), [METHOD_1, METHOD_2]):
    print(f'{method[10:]}: {v/3600:.2f}')

# Plot comparison of methods
plt.figure(figsize=[WIDTH, WIDTH], layout='constrained')
plt.xscale('log')
plt.yscale('log')

# Create lists of the dictionaries, sorted on dataset name, to ensure runtimes for the same datasets are compared
method_1_runtimes = [
    x[1] for x in sorted(method_1_runtimes.items()) if x[0] in method_2_runtimes
]
method_2_runtimes = [x[1] for x in sorted(method_2_runtimes.items())]

plt.scatter(
    method_2_runtimes, method_1_runtimes, s=SIZE, color='k', label='UCR data set'
)

# Add lines for equality, 10x less, and 100x less runtime
plt.plot([0.05, 250000], [0.05, 250000], 'k', lw=SIZE, label='Equality')
plt.plot([0.5, 250000], [0.05, 25000], 'k--', lw=SIZE, label='10x Faster')
plt.plot([5, 250000], [0.05, 2500], 'k:', lw=SIZE, label='100x Faster')

# Aesthetics
plt.xlim([10e-3, 10e5])
plt.ylim([10e-3, 10e5])
plt.xlabel(METHOD_1[10:])
plt.ylabel(METHOD_2[10:])
plt.legend()

plt.savefig(join('figs', '1 runtime.eps'))
plt.savefig(join('figs', '1 runtime.png'))
plt.close()
