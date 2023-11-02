from collections import defaultdict
from os.path import join

import matplotlib.pyplot as plt

from plot import WIDTH

FILE = 'e1_runtime.csv'
SIZE = 0.3

cpp_miner_runtimes = {}
ostinato_runtimes = {}
runtimes = defaultdict(float)
with open(FILE) as fp:
    for line in fp.readlines():
        row = line[:-1].split(',')
        method, dataset, runtime = row
        runtime = float(runtime)
        runtimes[method] += runtime

        if method == 'benchmark_cpp_miner':
            cpp_miner_runtimes[dataset] = float(runtime)
        elif method == 'benchmark_ostinato':
            ostinato_runtimes[dataset] = float(runtime)

# Print total run times
for k, v in runtimes.items():
    print(f'{k[10:]}: {v/3600:.2f}')

# Plot comparison of FRM-Miner (cpp implementation) and stumpy
plt.figure(figsize=[WIDTH, WIDTH], layout='constrained')
plt.xscale('log')
plt.yscale('log')

# Create lists of the dictionaries, sorted on dataset name, to ensure runtimes for the same datasets are compared
cpp_miner_runtimes = [x[1] for x in sorted(cpp_miner_runtimes.items())]
ostinato_runtimes = [x[1] for x in sorted(ostinato_runtimes.items())]

plt.scatter(ostinato_runtimes, cpp_miner_runtimes, s=SIZE, color='k', label='UCR data set')

# Add lines for equality, 10x less, and 100x less runtime
plt.plot([0.05, 250000], [0.05, 250000], 'k', lw=SIZE, label='Equality')
plt.plot([0.5, 250000], [0.05, 25000], 'k--', lw=SIZE, label='10x Faster')
plt.plot([5, 250000], [0.05, 2500], 'k:', lw=SIZE, label='100x Faster')

# Aesthetics
plt.xlim([10e-3, 10e5])
plt.ylim([10e-3, 10e5])
plt.xlabel('Ostinato')
plt.ylabel('FRM-Miner')
plt.legend()

plt.savefig(join('figs', f'1 runtime.eps'))
plt.close()
