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

from frm import Miner

FILE = "e5_runtime.csv"
FOLDER = "UCRArchive_2018"
FILES = listdir(FOLDER)
PARTITIONS = ["TRAIN", "TEST"]

# Parameters for FRM-Miner
MINSUP = 0.3
SEGLEN = 2
ALPHA = 4


def main():
    # Get already calculated combinations from file
    with open(FILE) as fp:
        seen = [row.split(",")[:2] for row in fp.readlines()]

    # Benchmark different miners using multiprocessing
    MINERS = [benchmark_miner_2]
    with Pool(processes=32, maxtasksperchild=1) as p:
        c = product(MINERS, FILES)
        unseen = [(m, n) for (m, n) in c if [f"{m.__name__}", n] not in seen]
        c = product(MINERS, FILES)
        print(f"Already done {len(list(c)) - len(unseen)} settings...")
        print(f'"just" {len(unseen)} to go!')
        p.starmap(benchmark, unseen)


def benchmark(miner, name):
    print(f"{miner.__name__}: {name}...")
    data = get_data(name)
    start = perf_counter()
    miner(data)
    end = perf_counter()
    with open(FILE, "a") as fp:
        fp.write(f"{miner.__name__},{name},{end - start}\n")
    print(f"{miner.__name__}: {name} done!")


def benchmark_miner_2(data):
    miner = Miner(MINSUP, SEGLEN, ALPHA)
    miner.mine(data)


def get_data(name):
    data = []
    for part in PARTITIONS:
        with open(join(FOLDER, name, f"{name}_{part}.tsv")) as f:
            for row in f:
                # Split data on tabs and parse to floats
                data.append(
                    [float(x) for x in row.strip("\n").split("\t")[1:] if x != "NaN"]
                )

    return data


if __name__ == "__main__":
    print("job start")
    main()
