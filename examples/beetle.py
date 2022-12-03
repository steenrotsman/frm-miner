from os.path import join

import numpy as np

from utils import parse, plot
from motifminer import Miner

def main():
    args = parse()
    data = np.loadtxt(join('data', 'BeetleFly.tsv'), delimiter='\t')[:, 1:]
    
    mm = Miner(data, args.min_sup, args.w, args.a, args.l)
    motifs = mm.mine_motifs()

    if args.plot:
        plot(data, motifs)


if __name__ == '__main__':
    main()