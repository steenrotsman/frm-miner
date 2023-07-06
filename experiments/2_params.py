from time import perf_counter
from itertools import product
from multiprocessing import Pool

from frm._frm_py.miner import Miner

from stocks import get_stocks

FILE = '2_params.csv'


# Parameter ranges for FRM-Miner
MINSUP = [0.1, 0.3, 0.5, 0.7, 0.9]
SEGLEN = [1, 2, 5, 10, 20, 50, 100]
ALPHABET = [2, 3, 4, 5, 6, 7, 8, 9, 10]


data = get_stocks()

def main():

    # Get already calculated combinations from file
    with open(FILE) as fp:
        seen = [tuple(row.split(',')[:3]) for row in fp.readlines()]

    with Pool(12) as p:
        settings = [setting for setting in product(MINSUP, SEGLEN, ALPHABET) if setting not in seen]
        p.starmap(mine, settings)


def mine(minsup, seglen, alphabet):
    start = perf_counter()
    miner = Miner(minsup, seglen, alphabet)
    miner.mine(data)
    end = perf_counter()

    with open(FILE, 'a') as fp:
        fp.write(f'{minsup},{seglen},{alphabet},{end-start},{len(miner.motifs)}')


if __name__ == '__main__':
    main()
