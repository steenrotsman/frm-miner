"""
To replicate this experiment, add a copy of UCRArchive_2018 from https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
Results are saved to the file e1_memory.csv.

Data sets are supplied by the UCR archive as two .tsv files, a train set and a test set.
The train and test files are joined into one data set, on which the algorithms are applied.
Peak memory use on each data set is recorded.
"""

from e1_runtime import FILES, benchmark_miner_2_1, benchmark_miner_2_2, get_data
from memory_profiler import memory_usage

FILE = 'e1_memory.csv'


def main():
    # Get already calculated combinations from file
    with open(FILE) as fp:
        seen = [row.split(',')[:2] for row in fp.readlines()]

    # Benchmark different miners using multiprocessing
    MINERS = [benchmark_miner_2_1, benchmark_miner_2_2]
    unseen = [(m, n) for m in MINERS for n in FILES if [m.__name__, n] not in seen]
    for miner, name in unseen:
        benchmark(miner, name)


def benchmark(miner, name):
    print(f'{miner.__name__}: {name}...')
    data = get_data(name)
    peak = memory_usage((miner, (data,)), max_usage=True)
    with open(FILE, 'a') as fp:
        fp.write(f'{miner.__name__},{name},{peak}\n')
    print(f'{miner.__name__}: {name} done!')


if __name__ == '__main__':
    main()
