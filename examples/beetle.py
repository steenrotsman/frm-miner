from os.path import join

from utils import parse, plot
import numpy as np
import tensorflow as tf

from motifminer import Miner

def main():
    args = parse()
    data = np.loadtxt(join('data', 'BeetleFly.tsv'), delimiter='\t')[:, 1:]
    data = tf.constant(data, dtype=tf.float32)
    
    mm = Miner(data, args.min_sup, args.w, args.a, args.l, args.k, args.m)
    motifs = mm.mine_motifs()

    if args.plot:
        plot(data, motifs, args.w, args.a)


if __name__ == '__main__':
    main()