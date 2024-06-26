"""
To replicate this experiment, add a copy of UCRArchive_2018 from https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
Results are saved to the file e1_runtime.csv.

Data sets are supplied by the UCR archive as two .tsv files, a train set and a test set.
The train and test files are joined into one data set, on which the algorithms are applied.
Run time on each data set is recorded.
"""

from itertools import product
from multiprocessing import Pool
from os import listdir
from os.path import join
from time import perf_counter

import numpy as np
import stumpy

from frm import Miner

FILE = 'e1_runtime.csv'
FOLDER = 'UCRArchive_2018'
FILES = listdir(FOLDER)
PARTITIONS = ['TRAIN', 'TEST']

# Parameters for FRM-Miner
MINSUP = 0.3
SEGLEN = 1
ALPHA = 4


def main():
    # Get already calculated combinations from file
    with open(FILE) as fp:
        seen = [row.split(',')[:2] for row in fp.readlines()]

    # Benchmark different miners using multiprocessing
    MINERS = [benchmark_stumpy]
    # with Pool(processes=32, maxtasksperchild=1) as p:
    c = product(MINERS, FILES, range(1, 11))
    unseen = [(m, n, s) for (m, n, s) in c if [f'{m.__name__}_{s}', n] not in seen]
    c = product(MINERS, FILES, range(1, 11))
    print(f'Already done {len(list(c)) - len(unseen)} settings...')
    print(f'"just" {len(unseen)} to go!')
    # p.starmap(benchmark, unseen)
    for u in unseen:
        benchmark(*u)


def benchmark(miner, name, seglen):
    print(f'{miner.__name__}_{seglen}: {name}...')
    data = get_data(name)
    start = perf_counter()
    miner(data, seglen)
    end = perf_counter()
    with open(FILE, 'a') as fp:
        fp.write(f'{miner.__name__}_{seglen},{name},{end-start}\n')
    print(f'{miner.__name__}_{seglen}: {name} done!')


def benchmark_miner_2(data, seglen):
    miner = Miner(MINSUP, seglen, ALPHA)
    miner.mine(data)


def benchmark_stumpy(data, seglen):
    data, length = get_length(data, seglen)
    stumpy.ostinato(data, length)


def get_data(name):
    data = []
    for part in PARTITIONS:
        with open(join(FOLDER, name, f'{name}_{part}.tsv')) as f:
            for row in f:
                # Split data on tabs and parse to floats
                data.append(
                    [float(x) for x in row.strip('\n').split('\t')[1:] if x != 'NaN']
                )

    return data


def get_length(data, seglen):
    # Shrink time series length with PAA
    data = [paa(ts, seglen) for ts in data]

    # The consensus motif for lengths shorter than 3 is meaningless
    data = [ts for ts in data if len(ts) >= 3]

    # In principle, the length should be 1/10th of the length of the longest ts
    lens = list(map(len, data))
    length = max(lens) // 10

    # If there are very short time series, set the length to half of the shortest
    if any(tslen < length for tslen in lens):
        length = min(lens) // 2

    # The consensus motif for lengths shorter than 3 is meaningless
    length = max(length, 3)

    return data, length


def paa(series, seglen):
    if too_short := (len(series) % seglen):
        append = [np.mean(series[-too_short:])] * (seglen - too_short)
        series = np.append(series, append)

    segments = np.reshape(series, (-1, seglen))
    return np.mean(segments, axis=1)


if __name__ == '__main__':
    main()
