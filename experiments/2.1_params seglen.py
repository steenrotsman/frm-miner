from time import perf_counter

import matplotlib.pyplot as plt
from frm._frm_py.miner import Miner

from stocks import get_stocks
from plot import plot_motifs

# Parameters for FRM-Miner
MINSUP = 0.3
SEGLEN = [10, 50, 100]
ALPHABET = 5


def main():
    data = get_stocks()
    fig, axss = plt.subplots(3, 3)
    for seglen, axs in zip(SEGLEN, axss):
        miner = Miner(MINSUP, seglen, ALPHABET)
        motifs = miner.mine_motifs(data)
        print(len(motifs))
        # plot_motifs(fig, axs, motifs, ALPHABET)
    plt.show()

if __name__ == '__main__':
    main()
