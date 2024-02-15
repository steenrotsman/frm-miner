from itertools import product
from multiprocessing import Pool
from time import perf_counter

from e2_stocks import get_stocks

from frm import Miner

FILE = 'e2_params.csv'


# Parameter ranges for FRM-Miner
MINSUP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEGLEN = [10, 15, 20, 25, 30, 35, 40, 45, 50]
ALPHABET = [2, 3, 4, 5, 6, 7, 8, 9, 10]
N = 5

data = get_stocks()


def main():
    # Get already calculated combinations from file
    with open(FILE) as fp:
        seen = []
        for row in fp.readlines():
            i, m, s, a = tuple(row.split(',')[:4])
            seen.append((int(i), float(m), int(s), int(a)))

    with Pool(16, maxtasksperchild=1) as p:
        settings = [
            s for s in product(range(N), MINSUP, SEGLEN, ALPHABET) if s not in seen
        ]
        p.starmap(mine, settings)


def mine(i, minsup, seglen, alphabet):
    print(f'{i},{minsup},{seglen},{alphabet}...')
    start = perf_counter()
    miner = Miner(minsup, seglen, alphabet)
    miner.mine(data)
    end = perf_counter()
    print(f'{i},{minsup},{seglen},{alphabet} done!')

    with open(FILE, 'a') as fp:
        fp.write(f'{i},{minsup},{seglen},{alphabet},{end-start},{len(miner.motifs)}\n')


if __name__ == '__main__':
    main()
