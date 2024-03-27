from collections import defaultdict
from itertools import product
from os.path import join
from statistics import fmean
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from e1_runtime import benchmark_miner_2_1
from e4_scalability_ucr import get_ucr_results
from patterns import profile_memory_peak
from plot import remove_spines

TIME_FILE = 'e4_scalability sim.csv'
SPACE_FILE = 'e4_scalability sim memory.csv'
NAME = '4 scalability sim'

LENGTHS = [10**i for i in range(1, 5)]
ROWS = [10**i for i in range(1, 5)]
INJECT = 40
ITER = 10
PLOT = False

RNG = np.random.default_rng(0)


def main():
    experiment(TIME_FILE, 'Seconds', PLOT)
    experiment(SPACE_FILE, 'Bytes', PLOT)


def experiment(file, measure, plot):
    seen = get_seen(file)
    for setting in product(LENGTHS, ROWS, range(ITER)):
        if setting not in seen:
            simulation(*setting, file, measure)
    results = get_results(file)
    if plot:
        plot_results(*results, NAME, measure)


def get_seen(file):
    with open(file) as fp:
        seen = []
        for row in fp.readlines():
            length, rows, i = tuple(row.split(',')[:3])
            seen.append((int(length), int(rows), int(i)))
    return seen


def simulation(length, rows, iter, file, measure):
    name = f'{length} {rows} {iter} ({measure})'
    print(f'{name}...')
    data = get_data(length, rows)

    if measure == 'Bytes':
        result = profile_memory_peak(data, 2, seglen=1) / 1e6
    else:
        start = perf_counter()
        benchmark_miner_2_1(data)
        end = perf_counter()
        result = end - start

    with open(file, 'a') as fp:
        fp.write(f'{length},{rows},{iter},{result}\n')
    print(f'{name} done!')


def get_data(length, rows):
    data = RNG.standard_normal((rows, length))
    motif_length = length // 10
    motif = RNG.standard_normal(motif_length)
    inject = RNG.integers(1, length - motif_length, rows)
    inject *= RNG.integers(0, 100, rows) < INJECT
    for i, idx in enumerate(inject):
        if idx:
            data[i][idx : idx + motif_length] = motif
    return data


def get_results(file):
    lengths = defaultdict(list)
    rows = defaultdict(list)
    total = defaultdict(list)

    with open(file) as fp:
        for row in fp.readlines():
            length, row, iter, runtime = tuple(row.split(','))
            length, row, runtime = int(length), int(row), float(runtime)
            lengths[length].append(runtime)
            rows[row].append(runtime)
            total[row * length].append(runtime)

    lengths = {size: fmean(x) for size, x in lengths.items()}
    rows = {size: fmean(x) for size, x in rows.items()}
    total = {size: fmean(x) for size, x in total.items()}

    return lengths, rows, total


def plot_results(lengths, rows, total, name, measure, marker='.', ls='-'):
    fig, axs = plt.subplots(ncols=3)

    # Plot the lines for number of rows, row length, and total data size
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    params = {
        'marker': marker,
        'ls': ls,
        'ms': 3,
        'lw': 1,
        'c': cycle[0],
        'label': 'Simulation',
    }
    xlabels = ['Time series quantity', 'Time series length', 'Total database size']
    for i in range(2):
        for ax, res, xlabel in zip(axs, [rows, lengths, total], xlabels):
            data = list(zip(*sorted(res.items())))
            ax.plot(*data, **params)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(xlabel)
            remove_spines(ax, False)
            yticks = calculate_ticks(data)
            ax.set_yticks(yticks)
            ax.tick_params(axis='y', which='minor', left=False)

        # Update parameters to UCR archive for second round
        if not i:
            lengths, rows, total = get_ucr_results(measure)
        params['ls'] = ''
        params['c'] = 'k'
        params['label'] = 'UCR data set'
        axs[0].set_ylabel(measure)

    plt.savefig(join('figs', f'{name} {measure}.eps'))
    plt.savefig(join('figs', f'{name} {measure}.png'))
    plt.close()


def calculate_ticks(data):
    x_data, y_data = data
    ymin = 10 ** round(np.log10(np.min(y_data)))
    ymax = 10 ** round(np.log10(np.max(y_data)))
    yticks = [ymin, ymax]

    return yticks


if __name__ == '__main__':
    main()
