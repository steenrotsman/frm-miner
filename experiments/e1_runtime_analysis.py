from collections import defaultdict
from os.path import join

import matplotlib.pyplot as plt
from plot import WIDTH

FILE = 'e1_runtime.csv'
SIZE = 0.3
X_METHOD = 'benchmark_py_miner_new'
Y_METHOD = 'benchmark_cpp_miner_old'
X_NAME = 'New Python implementation'
Y_NAME = 'Previous C++ implementation'

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
plt.figure(figsize=[WIDTH, WIDTH], layout='constrained')
plt.xscale('log')
plt.yscale('log')

# Create lists of the dictionaries, sorted on dataset name, to ensure runtimes for the same datasets are compared
x_method_runtimes = [x[1] for x in sorted(x_method_runtimes.items())]
y_method_runtimes = [x[1] for x in sorted(y_method_runtimes.items())]

plt.scatter(
    y_method_runtimes, x_method_runtimes, s=SIZE, color='k', label='UCR data set'
)

# Add lines for equality, 10x less, and 100x less runtime
plt.plot([0.05, 250000], [0.05, 250000], 'k', lw=SIZE, label='Equality')
plt.plot([0.5, 250000], [0.05, 25000], 'k--', lw=SIZE, label='10x Faster')
plt.plot([5, 250000], [0.05, 2500], 'k:', lw=SIZE, label='100x Faster')

# Aesthetics
plt.xlim([10e-3, 10e5])
plt.ylim([10e-3, 10e5])
plt.xlabel(X_NAME)
plt.ylabel(Y_NAME)
plt.legend()

plt.savefig(join('figs', '1 runtime.eps'))
plt.savefig(join('figs', '1 runtime.png'))
plt.close()
