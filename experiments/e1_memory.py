"""
To replicate this experiment, add a copy of UCRArchive_2018 from https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
Results are saved to the file e1_memory.csv.

Data sets are supplied by the UCR archive as two .tsv files, a train set and a test set.
The train and test files are joined into one data set, on which the algorithms are applied.
Peak memory use on each data set is recorded.
"""

from multiprocessing import Pool

from e1_runtime import FILES, get_data
from patterns import profile_memory_peak

FILE = "e1_memory.csv"


def main():
    # Get already calculated combinations from file
    with open(FILE) as fp:
        seen = [row.split(",")[:2] for row in fp.readlines()]

    # Benchmark different miners using multiprocessing
    unseen = [(frm, n) for frm in (1, 2) for n in FILES if [str(frm), n] not in seen]
    with Pool(8, maxtasksperchild=1) as p:
        p.starmap(benchmark, unseen)


def benchmark(frm, name):
    print(f"FRM-Miner {frm}.0: {name}...")
    data = get_data(name)
    peak = profile_memory_peak(data, frm) / 1e6
    with open(FILE, "a") as fp:
        fp.write(f"{frm},{name},{peak}\n")
    print(f"FRM-Miner {frm}.0: {name} done!")


if __name__ == "__main__":
    main()
