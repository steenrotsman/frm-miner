"""
To replicate this experiment, add a copy of UCRArchive_2018 from https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
Results are saved to the file e1_runtime.csv.

Data sets are supplied by the UCR archive as two .tsv files, a train set and a test set.
The train and test files are joined into one data set, one which the algorithms are applied.
"""
from os import listdir
from os.path import join
from time import perf_counter
from multiprocessing import Pool

import stumpy

from frm import Miner as CppMiner
from frm._frm_py.miner import Miner as PyMiner
from ostinato import ostinato

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
    MINERS = [benchmark_py_miner, benchmark_cpp_miner]
    with Pool(processes=1, maxtasksperchild=1) as p:
        p.starmap(benchmark, [(m, n) for m in MINERS for n in FILES if [m.__name__, n] not in seen])


def benchmark(miner, name):
    print(f'{miner.__name__}: {name}...')
    data = get_data(name)
    start = perf_counter()
    miner(data)
    end = perf_counter()
    with open(FILE, 'a') as fp:
        fp.write(f'{miner.__name__},{name},{end-start}\n')
    print(f'{miner.__name__}: {name} done!')


def benchmark_py_miner(data):
    py_miner = PyMiner(MINSUP, SEGLEN, ALPHABET)
    py_miner.mine(data)


def benchmark_cpp_miner(data):
    cpp_miner = CppMiner(MINSUP, SEGLEN, ALPHABET)
    cpp_miner.mine(data)


def benchmark_stumpy(data):
    length = max(max(map(len, data)) // 10, 3)
    stumpy.ostinato(data, length)


def benchmark_ostinato(data):
    length = max(max(map(len, data)) // 10, 3)
    ostinato(data, length)


def get_data(name):
    data = []
    for part in PARTITIONS:
        with open(join(FOLDER, name, f'{name}_{part}.tsv')) as f:
            for row in f:
                # Split data on tabs and parse to floats
                data.append([float(x) for x in row.strip('\n').split('\t')[1:] if x != 'NaN'])

    return data


if __name__ == '__main__':
    main()
