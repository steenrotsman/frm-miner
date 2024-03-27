from collections import defaultdict
from math import ceil, floor, log10
from os.path import join

import matplotlib.pyplot as plt
from plot import WIDTH

SIZE = 3

SETTINGS = [
    # file, x_method, y_method, x_name, y_name, lines, fn
    (
        'e1_runtime.csv',
        'benchmark_py_miner_1_2',
        'benchmark_miner_2_2',
        'FRM-Miner 1.0',
        'FRM-Miner 2.0',
        3,
        'frm runtime',
    ),
    (
        'e1_runtime.csv',
        'benchmark_stumpy',
        'benchmark_miner_2_1',
        'Ostinato',
        'FRM-Miner',
        4,
        'ostinato runtime',
    ),
    (
        'e1_memory.csv',
        'benchmark_miner_1_2',
        'benchmark_miner_2_2',
        'FRM-Miner 1.0',
        'FRM-Miner 2.0',
        3,
        'frm memory',
    ),
]


def main():
    for file, *settings in SETTINGS:
        results = get_results(file)
        print_results(results)
        plot_results(results, *settings)


def get_results(file):
    runtimes = defaultdict(dict)
    with open(file) as fp:
        for line in fp.readlines():
            row = line[:-1].split(',')
            method, dataset, runtime = row
            runtime = float(runtime)
            runtimes[method][dataset] = runtime
    return runtimes


def print_results(runtimes):
    for method, runtime in runtimes.items():
        print(f'{method[10:]}: {sum(runtime.values())/3600:.2f}')


def plot_results(runtimes, x_method, y_method, x_name, y_name, lines, fn):
    plt.figure(figsize=[WIDTH / 1.5, WIDTH / 1.5])
    plt.xscale('log')
    plt.yscale('log')

    # Create lists of the dictionaries, sorted on dataset name, to ensure runtimes for the same datasets are compared
    x_method_runtimes = [x[1] for x in sorted(runtimes[x_method].items())]
    y_method_runtimes = [x[1] for x in sorted(runtimes[y_method].items())]

    lim = [
        10 ** floor(log10(min(x_method_runtimes + y_method_runtimes))),
        10 ** ceil(log10(max(x_method_runtimes + y_method_runtimes))),
    ]
    plt.scatter(
        x_method_runtimes, y_method_runtimes, s=SIZE, color='k', label='UCR data set'
    )

    # Add lines for equality, and orders of magnitude difference
    start = 1.5 * lim[0]
    end = 0.7 * lim[1]

    for i, line in enumerate(['k', 'k--', 'k:', 'k-.'][:lines]):
        factor = 10**i
        label = f'{factor}x Difference'
        if not i:
            label = 'Equality'
        plt.plot([start * factor, end], [start, end / factor], line, label=label)

    # Aesthetics
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()

    plt.savefig(join('figs', f'1 {fn}.eps'))
    plt.savefig(join('figs', f'1 {fn}.png'))
    plt.close()


if __name__ == '__main__':
    main()
