from os import listdir
from os.path import join

import numpy as np

from utils import parse, plot
from motifminer import Miner
from motifminer.preprocessing import sax

COLUMN = 'AU45_r'

def main():
    args = parse()
    data = get_data()

    mm = Miner(data, args.min_sup, args.w, args.a, args.l, args.k, args.m)
    motifs = mm.mine_motifs()

    if args.plot:
        plot(data, motifs)


def get_data():
    try:
        data = np.load(join('data', f'{COLUMN}.npy'))
    except FileNotFoundError:
        path = join('data', 'tetris')
        files = listdir(path)

        cols = np.loadtxt(join(path, files[0]), delimiter=',', max_rows=1, dtype='str')
        col = np.where(cols == COLUMN)[0][0]

        data = []
        for file in files:
            row = np.loadtxt(join(path, file), delimiter=',', skiprows=1, usecols=col, max_rows=18270)
            data.append(row)

        data = np.array(data)
        np.save(join('data', COLUMN), data)
    
    return data


if __name__ == '__main__':
    main()