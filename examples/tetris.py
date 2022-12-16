from os import listdir
from os.path import join

from utils import parse, plot
import numpy as np
import tensorflow as tf

from motifminer import Miner
from motifminer.preprocessing import standardize

COLUMN = 'AU17_r'

def main():
    args = parse()
    data = get_data()

    mm = Miner(data, args.min_sup, args.w, args.a, args.l, args.o, args.k, args.m)
    motifs = mm.mine_motifs()

    if args.plot:
        plot(data, motifs, args.w, args.a)


def get_data():
    try:
        data = np.load(join('data', f'{COLUMN}.npy'), allow_pickle=True)
        data = tf.ragged.constant(data, dtype=tf.float32)
    except FileNotFoundError:
        path = join('data', 'tetris')
        files = listdir(path)

        cols = np.loadtxt(join(path, files[0]), delimiter=',', max_rows=1, dtype='str')
        col = np.where(cols == COLUMN)[0][0]

        data = []
        for file in files:
            row = np.loadtxt(join(path, file), delimiter=',', skiprows=1, usecols=col)
            data.append(row)

        data = tf.ragged.constant(data, dtype=tf.float32)
        np.save(join('data', COLUMN), data.numpy())
    
    return data


if __name__ == '__main__':
    main()