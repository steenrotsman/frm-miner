from collections import defaultdict
from statistics import fmean

from e1_memory import FILE as MEMORY_FILE
from e1_runtime import FILE as RUNTIME_FILE
from e1_runtime import get_data
from e4_scalability_sim import plot_results

NAME = '4 scalability ucr'


def main():
    for file, measure in zip([RUNTIME_FILE, MEMORY_FILE], ['Seconds', 'Bytes']):
        lengths = defaultdict(list)
        rows = defaultdict(list)
        total = defaultdict(list)

        with open(file) as fp:
            for row in fp.readlines():
                experiment, name, runtime = tuple(row.split(','))
                if 'benchmark_miner' not in experiment:
                    continue
                runtime = float(runtime)
                data = get_data(name)
                length = fmean(map(len, data))
                row = len(data)

                lengths[length].append(runtime)
                rows[row].append(runtime)
                total[row * length].append(runtime)

        lengths = {size: fmean(x) for size, x in lengths.items()}
        rows = {size: fmean(x) for size, x in rows.items()}
        total = {size: fmean(x) for size, x in total.items()}

        plot_results(lengths, rows, total, NAME, measure, ls='')


if __name__ == '__main__':
    main()
