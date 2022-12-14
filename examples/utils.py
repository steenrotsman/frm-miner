import sys
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

sys.path.append('.')
sys.path.append('..')
from motifminer.preprocessing import breakpoints

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

def plot(data, motifs, w, a):
    import matplotlib.pyplot as plt
    
    for motif in motifs:
        print(motif.pattern)
        plt.plot(tf.transpose(motif.matches), color='black', linewidth=0.1)
        plt.plot(motif.representative, color='blue')
        plt.hlines(breakpoints[a], 0, len(motif.representative), colors='black', linewidth=0.3)
        plt.ylim(-3, 3)
        plt.show()
    
        for k, m in motif.match_indexes.items():
            x = list(range(len(data[k])))
            y = data[k]
            m[0], m[1] = m[0] * w, m[1] * w
            plt.plot(x[:m[0]+1], y[:m[0]+1], linewidth=0.5, color='black')
            plt.plot(x[m[0]:m[1]+1], y[m[0]:m[1]+1], linewidth=2, color='blue')
            plt.plot(x[m[1]:], y[m[1]:], linewidth=0.5, color='black')
        plt.show()
