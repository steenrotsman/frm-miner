from collections import defaultdict
from os.path import join

import matplotlib.pyplot as plt
from plot import WIDTH

FILE = 'e1_runtime.csv'
SIZE = 3
X_METHOD = 'benchmark_stumpy'
Y_METHOD = 'benchmark_py_miner_new'
X_NAME = 'Ostinato'
Y_NAME = 'FRM-Miner'

x_method_runtimes = {}
y_method_runtimes = {}
runtimes = defaultdict(float)
with open(FILE) as fp:
    for line in fp.readlines():
        row = line[:-1].split(',')
        method, dataset, runtime = row
        runtime = float(runtime)
        runtimes[method] += runtime

        if method == X_METHOD:
            x_method_runtimes[dataset] = float(runtime)
        elif method == Y_METHOD:
            y_method_runtimes[dataset] = float(runtime)

# Print total run times
for method, runtime in runtimes.items():
    print(f'{method[10:]}: {runtime/3600:.2f}')

# Plot comparison of methods
plt.figure(figsize=[WIDTH / 1.5, WIDTH / 1.5])
plt.xscale('log')
plt.yscale('log')

# Create lists of the dictionaries, sorted on dataset name, to ensure runtimes for the same datasets are compared
x_method_runtimes = [x[1] for x in sorted(x_method_runtimes.items())]
y_method_runtimes = [x[1] for x in sorted(y_method_runtimes.items())]

plt.scatter(
    x_method_runtimes, y_method_runtimes, s=SIZE, color='k', label='UCR data set'
)

# Add lines for equality, 10x less, and 100x less runtime
plt.plot([0.05, 250000], [0.05, 250000], 'k', label='Equality')
plt.plot([0.5, 250000], [0.05, 25000], 'k--', label='10x Faster')
plt.plot([5, 250000], [0.05, 2500], 'k:', label='100x Faster')

# Aesthetics
plt.xlim([10e-3, 10e5])
plt.ylim([10e-3, 10e5])
plt.xlabel(X_NAME)
plt.ylabel(Y_NAME)
plt.legend()

plt.savefig(join('figs', '1 runtime.eps'))
plt.savefig(join('figs', '1 runtime.png'))
plt.close()
