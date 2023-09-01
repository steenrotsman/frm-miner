from statistics import fmean
from collections import defaultdict

import matplotlib.pyplot as plt
from plot import remove_spines, COLORS
from e1_runtime import get_data, FILE


def main():
    length_runtimes = defaultdict(list)
    row_runtimes = defaultdict(list)
    total_runtimes = defaultdict(list)

    with open(FILE) as fp:
        for row in fp.readlines():
            experiment, name, runtime = tuple(row.split(','))
            if 'cpp' not in experiment:
                continue
            runtime = float(runtime)
            data = get_data(name)
            length = fmean(map(len, data))
            rows = len(data)

            length_runtimes[length].append(runtime)
            row_runtimes[rows].append(runtime)
            total_runtimes[rows*length].append(runtime)

    length_runtimes = {size: fmean(x) for size, x in length_runtimes.items()}
    row_runtimes = {size: fmean(x) for size, x in row_runtimes.items()}
    total_runtimes = {size: fmean(x) for size, x in total_runtimes.items()}

    fig, axs = plt.subplots(nrows=2,sharey='all', layout='constrained')

    # Plot the lines for number of rows, row length, and total data size
    axs[0].scatter(list(row_runtimes.keys()), list(row_runtimes.values()), s=1, color=COLORS[0], label='Number of Rows')
    axs[0].scatter(list(length_runtimes.keys()), list(length_runtimes.values()), s=1, color=COLORS[1], label='Row Length')
    axs[1].scatter(list(total_runtimes.keys()), list(total_runtimes.values()), s=1, color=COLORS[3], label='Total Length')

    # Set the x-axis and y-axis scales to logarithmic
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_ylim([0.0001, 100000])
    remove_spines(axs[0], False)
    remove_spines(axs[1], False)

    # Set the x-axis and y-axis labels
    # axs[0].set_yticks([0.01, 10000])
    # axs[1].set_yticks([0.01, 10000])
    axs[0].set_ylabel('Run Time (seconds)', labelpad=0).set_y(-0.75)
    axs[1].set_xlabel('Size')

    # Show the plot
    plt.savefig(f'figs/4 scalability ucr.eps')


if __name__ == '__main__':
    main()
