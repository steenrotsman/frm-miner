from itertools import product
from multiprocessing import Pool
from time import perf_counter

from e2_stocks import get_stocks

from frm import Miner

FILE = 'e2_params.csv'


# Parameter ranges for FRM-Miner
MINSUP = [0.1, 0.3, 0.5, 0.7, 0.9]
SEGLEN = [10, 20, 30, 40, 50]
ALPHABET = [2, 3, 4, 5, 6, 7, 8, 9, 10]
OMAX = [0.5, 0.6, 0.7, 0.8, 0.9]
N = 5

data = get_stocks()


def main():
    # Get already calculated combinations from file
    with open(FILE) as fp:
        seen = []
        for row in fp.readlines():
            i, m, s, a, o = row.split(',')[:5]
            seen.append((int(i), float(m), int(s), int(a), float(o)))

    with Pool(8, maxtasksperchild=1) as p:
        settings = [
            s
            for s in product(range(N), MINSUP, SEGLEN, ALPHABET, OMAX)
            if s not in seen
        ]
        p.starmap(mine, settings)


def mine(i, minsup, seglen, alpha, omax):
    print(f'{i},{minsup},{seglen},{alpha},{omax}...')
    start = perf_counter()
    miner = Miner(minsup, seglen, alpha, max_overlap=omax)
    miner.mine(data)
    end = perf_counter()
    print(f'{i},{minsup},{seglen},{alpha},{omax} done!')

    with open(FILE, 'a') as fp:
        time = end - start
        fp.write(f'{i},{minsup},{seglen},{alpha},{omax},{time},{len(miner.motifs)}\n')


if __name__ == '__main__':
    main()
