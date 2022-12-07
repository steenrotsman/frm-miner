import sys
sys.path.append('.')
sys.path.append('..')
import argparse

import numpy as np

def parse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Discover frequent motifs \
            of variable length in time series.')
    parser.add_argument(
        '--min-sup', default=0.5, type=float, nargs='?',
        help='set minimum support \
            (default: %(default)s)')
    parser.add_argument(
        '-w', default=8, type=int, nargs='?',
        help='set Piecewise Aggregate Approximation window size \
            (default: %(default)s)')
    parser.add_argument(
        '-a', default=7, type=int, nargs='?',
        help='set Symbolic Aggregate Approximation alphabet size \
            (default: %(default)s)')
    parser.add_argument(
        '-l', default=3, type=int, nargs='?',
        help='minimal length of patterns to mine \
            (default: %(default)s)')
    parser.add_argument(
        '-k', default=0, type=int, nargs='?',
        help='how many patterns to find the motifs of, finds all if 0 \
            (default: %(default)s)')
    parser.add_argument(
        '-m', action='store_true',
        help='mine maximal motifs (default: all frequent motifs).'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='plot the original timeseries and the motifs')
    args = parser.parse_args()

    return args

def plot(data, motifs):
    import matplotlib.pyplot as plt

    plt.plot(data.T)
    plt.show()

    for motif in motifs:
        plt.plot(np.array(motif.matches).T, color='black', linewidth=0.1)
        plt.plot(motif.representative, color='blue')
        plt.ylim(-3, 3)
        plt.show()
