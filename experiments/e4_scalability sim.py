from time import perf_counter
from itertools import product, starmap
from multiprocessing import Pool
from math import log10
from statistics import fmean
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from frm._frm_py.miner import Miner
from plot import remove_spines, COLORS

FILE = 'e4_scalability sim.csv'

MINSUP = 0.3
SEGLEN = 10
ALPHABET = 5

LENGTHS = [10 ** (i+1) for i in range(1, 5)]
ROWS = [10 ** (i+1) for i in range(1, 5)]
INJECT = 40
ITER = 10


def main():
    with open(FILE) as fp:
        seen = []
        for row in fp.readlines():
            length, rows, i= tuple(row.split(',')[:3])
            seen.append((int(length), int(rows), int(i)))

    # with Pool(1, maxtasksperchild=1) as p:
    #     p.starmap(simulation, [x for x in product(LENGTHS, ROWS, range(ITER)) if x not in seen and x[0]*x[1] < 10**9])

    length_runtimes = defaultdict(list)
    row_runtimes = defaultdict(list)
    total_runtimes = defaultdict(list)

    with open(FILE) as fp:
        for row in fp.readlines():
            length, rows, iter, runtime = tuple(row.split(','))
            length, rows, runtime = int(length), int(rows), float(runtime)
            length_runtimes[length].append(runtime)
            row_runtimes[rows].append(runtime)
            total_runtimes[rows*length].append(runtime)

    length_runtimes = {size: fmean(x) for size, x in length_runtimes.items()}
    row_runtimes = {size: fmean(x) for size, x in row_runtimes.items()}
    total_runtimes = {size: fmean(x) for size, x in total_runtimes.items()}

    fig, axs = plt.subplots(nrows=2,sharey='all', layout='constrained')

    # Plot the lines for number of rows, row length, and total data size
    axs[0].plot(list(row_runtimes.keys()), list(row_runtimes.values()), '.-', color=COLORS[0], label='Number of Rows')
    axs[0].plot(list(length_runtimes.keys()), list(length_runtimes.values()), '.-', color=COLORS[1], label='Row Length')
    axs[1].plot(list(total_runtimes.keys()), list(total_runtimes.values()), '.-', color=COLORS[3], label='Total Length')

    # Set the x-axis and y-axis scales to logarithmic
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_ylim([0.0001, 100000])
    remove_spines(axs[0], False)
    remove_spines(axs[1], False)

    # Set the x-axis and y-axis labels
    axs[0].set_yticks([0.01, 10000])
    axs[1].set_yticks([0.01, 10000])
    axs[0].set_ylabel('Run Time (seconds)', labelpad=0).set_y(-0.75)
    axs[1].set_xlabel('Size')

    # Show the plot
    plt.savefig(f'figs/4 scalability sim.eps')


def simulation(length, rows, iter):
    print(f'{length} {rows} {iter}...')
    data = np.random.random((rows, length))

    motif = np.random.random(length // 10)
    inject = np.random.randint(1, length - length//10, rows)
    inject *= np.random.randint(0, 100, rows) < INJECT
    for i, idx in enumerate(inject):
        if i:
            data[i][idx : idx + length//10] = motif

    # Mine motifs in data
    start = perf_counter()
    miner = Miner(MINSUP, SEGLEN, ALPHABET)
    miner.mine(data)
    end = perf_counter()

    with open(FILE, 'a') as fp:
        fp.write(f'{length},{rows},{iter},{end - start}\n')
    print(f'{length} {rows} {iter} done!')


if __name__ == '__main__':
    main()
