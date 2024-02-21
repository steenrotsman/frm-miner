"""
To replicate this experiment, add a copy of UCRArchive_2018 from https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
Results are saved to the file e1_runtime.csv.

Data sets are supplied by the UCR archive as two .tsv files, a train set and a test set.
The train and test files are joined into one data set, on which the algorithms are applied.
Run time on each data set is recorded.
"""

from multiprocessing import Pool
from os import listdir
from os.path import join
from time import perf_counter

import stumpy
from ostinato import ostinato

from frm import Miner

FILE = 'e1_runtime.csv'
FOLDER = 'UCRArchive_2018'
FILES = listdir(FOLDER)
PARTITIONS = ['TRAIN', 'TEST']

# Parameters for FRM-Miner
MINSUP = 0.3
SEGLEN = 1
ALPHABET = 5


def main():
    # Get already calculated combinations from file
    with open(FILE) as fp:
        seen = [row.split(',')[:2] for row in fp.readlines()]

    # Benchmark different miners using multiprocessing
    MINERS = [
        benchmark_miner,
        benchmark_stumpy,
        benchmark_ostinato,
    ]
    with Pool(processes=16, maxtasksperchild=1) as p:
        unseen = [(m, n) for m in MINERS for n in FILES if [m.__name__, n] not in seen]
        p.starmap(benchmark, unseen)


def benchmark(miner, name):
    print(f'{miner.__name__}: {name}...')
    data = get_data(name)
    start = perf_counter()
    miner(data)
    end = perf_counter()
    with open(FILE, 'a') as fp:
        fp.write(f'{miner.__name__},{name},{end-start}\n')
    print(f'{miner.__name__}: {name} done!')


def benchmark_miner(data):
    miner = Miner(MINSUP, SEGLEN, ALPHABET)
    miner.mine(data)


def benchmark_stumpy(data):
    data, length = get_length(data)
    stumpy.ostinato(data, length)


def benchmark_ostinato(data):
    data, length = get_length(data)
    ostinato(data, length)


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


def get_length(data):
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


if __name__ == '__main__':
    main()
