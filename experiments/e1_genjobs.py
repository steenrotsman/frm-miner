"""
To replicate this experiment, add a copy of UCRArchive_2018 from https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
Results are saved to the file e1_runtime.csv.

Data sets are supplied by the UCR archive as two .tsv files, a train set and a test set.
The train and test files are joined into one data set, on which the algorithms are applied.
Run time on each data set is recorded.
"""

from itertools import product
from os import listdir

SEEN_FILE = "e1_runtime.csv"
JOBS_FILE = "e1_jobs.csv"
FOLDER = "UCRArchive_2018"
FILES = listdir(FOLDER)
PARTITIONS = ["TRAIN", "TEST"]

# Parameters for FRM-Miner
MINSUP = 0.3
SEGLEN = 1
ALPHA = 4


def main():
    # Get already calculated combinations from file
    with open(SEEN_FILE) as fp:
        seen = [row.split(",")[:2] for row in fp.readlines()]

    # Benchmark different miners using multiprocessing
    MINERS = ["benchmark_miner_2", "benchmark_stumpy"]

    c = product(MINERS, FILES, range(1, 11))
    unseen = [(m, n, s) for (m, n, s) in c if [f"{m}_{s}", n] not in seen]

    with open(JOBS_FILE, "w") as fp:
        for m, n, s in sorted(unseen, key=lambda x: x[::-1]):
            fp.write(f"{m},{n},{s}\n")


if __name__ == "__main__":
    main()
