from os.path import join

from utils import parse, plot
import numpy as np
import tensorflow as tf

from motifminer import Miner
from motifminer.preprocessing import standardize

def main():
    args = parse()
    data = []
    with open(join('data', 'GesturePebbleZ1.tsv')) as f:
        for row in f:
            data.append(row.strip('\n').split('\t')[1:])
    data = tf.ragged.constant(data)
    data = tf.strings.to_number(data)
    data = standardize(data)
    
    mm = Miner(data, args.min_sup, args.w, args.a, args.l, args.k, args.m)
    motifs = mm.mine_motifs()

    if args.plot:
        plot(data, motifs, args.w, args.a)


if __name__ == '__main__':
    main()