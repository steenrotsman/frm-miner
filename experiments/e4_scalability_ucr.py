from collections import defaultdict
from statistics import fmean

from e1_memory import FILE as MEMORY_FILE
from e1_runtime import FILE as RUNTIME_FILE
from e1_runtime import get_data


def get_ucr_results(measure):
    file = RUNTIME_FILE if measure == 'Seconds' else MEMORY_FILE
    lengths = defaultdict(list)
    rows = defaultdict(list)
    total = defaultdict(list)

    with open(file) as fp:
        for row in fp.readlines():
            experiment, name, result = tuple(row.split(','))
            if experiment in ['benchmark_miner_2_2', '2']:
                result = float(result)
                data = get_data(name)
                length = fmean(map(len, data))
                row = len(data)

                lengths[length].append(result)
                rows[row].append(result)
                total[row * length].append(result)

    lengths = {size: fmean(x) for size, x in lengths.items()}
    rows = {size: fmean(x) for size, x in rows.items()}
    total = {size: fmean(x) for size, x in total.items()}
    return lengths, rows, total
