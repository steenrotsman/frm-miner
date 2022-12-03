from os.path import join

import numpy as np

from parser import parse
from motifminer import Miner

def main():
    args = parse()
    data = np.loadtxt(join('data', 'BeetleFly.tsv'), delimiter='\t')[:, 1:]
    
    mm = Miner(data, args.min_sup, args.w, args.a, args.l)
    motifs = mm.mine_motifs()

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(data.T)
        plt.show()

        for motif in motifs:
            plt.plot(motif[0].T)
            plt.ylim(-3, 3)
            plt.show()


if __name__ == '__main__':
    main()