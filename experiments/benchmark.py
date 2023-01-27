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

import tensorflow as tf
import numpy as np
import stumpy

from motifminer.miner import Miner

FILE = 'benchmark.csv'
FOLDER = 'UCRArchive_2018'
FILES = ['Mallat', 'OliveOil', 'ToeSegmentation1', 'InlineSkate', 'FaceAll']

PARTITIONS = ['TRAIN', 'TEST']

# Parameters for proposed algorithm
MIN_SUP = [0.5, 0.7, 0.9]
W = [4, 8, 16]
A = [5, 7, 9]
L = [3]
O = [0.7, 0.8, 0.9]

# Parameters for baseline algorithms
M = [25, 50, 100]


def main():
    # Get already calculated combinations from file
    combinations = []
    with open(FILE) as fp:
        for row in fp.readlines():
            combinations.append(row.split(',')[1])

    # Calculate and save run times to file
    with open(FILE, 'a') as fp:
        for name, data in get_data():
            benchmark_mm(data, name, fp, combinations)
            benchmark_ostinato([a.astype(np.float64) for a in data.numpy()], name, fp, combinations)


def get_data():
    for fn in FILES:
        data = []
        for part in PARTITIONS:
            with open(join(FOLDER, fn, f'{fn}_{part}.tsv')) as f:
                for row in f:
                    data.append(row.strip('\n').split('\t')[1:])
        data = tf.ragged.constant(data)
        data = tf.strings.to_number(data)
        yield fn, data


def benchmark_mm(data, name, fp, combinations):
    for min_sup, w, a, l, o in product(MIN_SUP, W, A, L, O):
        if (combination := f'mm_{min_sup}_{w}_{a}_{l}_{o}') in combinations:
            continue

        start = perf_counter()
        mm = Miner(data, min_sup, w, a, l, o, 0, False)
        mm.mine_motifs()
        end = perf_counter()

        fp.write(f'{name},{combination},{end-start}\n')


def benchmark_ostinato(data, name, fp, combinations):
    for m in M:
        if combination := f'ostinato_{m}' in combinations:
            continue

        start = perf_counter()
        stumpy.ostinato(data, m)
        end = perf_counter()

        fp.write(f'{name},{combination},{end-start}\n')


if __name__ == '__main__':
    main()
