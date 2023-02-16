"""
This module provides a runtime comparison of 5 data sets of the proposed method and existing methods.

Baseline methods that are compared against are:
- Ostinato

Data sets are supplied by the UCR archive as two .tsv files, a train set and a test set.
The train and test files are joined into one data set, for which the baseline is then mined

To replicate this experiment, add your own unmodified copy of UCRArchive_2018 found here https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
Results are saved to the file runtime.csv.
"""
from os.path import join
from itertools import product
from time import perf_counter
from collections import defaultdict

import stumpy

from motifminer.miner import Miner

FILE = 'benchmark.csv'
FOLDER = 'UCRArchive_2018'
FILES = ['Mallat', 'OliveOil', 'ToeSegmentation1', 'InlineSkate', 'FaceAll']
ITER = 10
PARTITIONS = ['TRAIN', 'TEST']

# Parameters for proposed algorithm
MIN_SUP = [0.5, 0.7, 0.9]
SEGMENT = [4, 8, 16]
ALPHABET = [5, 7, 9]
MIN_LEN = [3]
MAX_OVERLAP = [0.7, 0.8, 0.9]

# Parameters for baseline algorithms
LENGTH = [25, 50, 100]


def main():
    # Get already calculated combinations from file
    seen = defaultdict(list)
    with open(FILE) as fp:
        for row in fp.readlines():
            fields = row.split(',')
            seen[fields[0]].append(fields[1])

    # Calculate and save run times to file
    with open(FILE, 'a') as fp:
        for name, data in get_data():
            benchmark_mm(data, name, fp, seen[name])
            benchmark_ostinato(data, name, fp, seen[name])


def get_data():
    for fn in FILES:
        data = []
        for part in PARTITIONS:
            with open(join(FOLDER, fn, f'{fn}_{part}.tsv')) as f:
                for row in f:
                    # Split data on tabs
                    data.append(row.strip('\n').split('\t')[1:])

        # Parse data to floats
        data = [[float(x) for x in row] for row in data]

        yield fn, data


def benchmark_mm(data, name, fp, seen):
    for min_sup, s, a, l, o, i in product(MIN_SUP, SEGMENT, ALPHABET, MIN_LEN, MAX_OVERLAP, range(ITER)):
        if (combination := f'mm_{min_sup}_{s}_{a}_{l}_{o}_{i}') in seen:
            continue

        start = perf_counter()
        mm = Miner(data, min_sup, s, a, l, o)
        mm.mine_motifs()
        end = perf_counter()

        fp.write(f'{name},{combination},{end-start}\n')


def benchmark_ostinato(data, name, fp, seen):
    for m, i in product(LENGTH, range(ITER)):
        if (combination := f'ostinato_{m}_{i}') in seen:
            continue

        start = perf_counter()
        stumpy.ostinato(data, m)
        end = perf_counter()

        fp.write(f'{name},{combination},{end-start}\n')


if __name__ == '__main__':
    main()
