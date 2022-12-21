import sys
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import stumpy

sys.path.append('.')
sys.path.append('..')
from motifminer.preprocessing import breakpoints

YMIN, YMAX = -3, 3

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
        '-o', default=0.9, type=float, nargs='?',
        help='maximal fraction of patterns contained in longer patters \
            (default: %(default)s)')
    parser.add_argument(
        '-k', default=0, type=int, nargs='?',
        help='how many patterns to find the motifs of, finds all if 0 \
            (default: %(default)s)')
    parser.add_argument(
        '-m', action='store_true',
        help='mine maximal motifs (default: all frequent motifs).')
    parser.add_argument(
        '--plot', action='store_true',
        help='plot the original timeseries and the motifs')
    args = parser.parse_args()

    return args

def plot(data, motifs, w, a):
    for motif in motifs:
        # Plot motif occurences and representative
        plt.plot(tf.transpose(motif.matches), 'k', lw=0.1)
        plt.plot(motif.representative, 'b', lw=1)
        plt.hlines(breakpoints[a], 0, len(motif.representative), 'k', lw=0.3)
        plt.ylim(YMIN, YMAX)
        plt.show()
    
        # Plot motif occurences in time series
        for k, m in motif.match_indexes.items():
            x = list(range(len(data[k])))
            y = data[k]
            m[0], m[1] = m[0] * w, m[1] * w
            plt.plot(x[:m[0]+1], y[:m[0]+1], 'k', lw=0.5)
            plt.plot(x[m[0]:m[1]+1], y[m[0]:m[1]+1], 'b', lw=1.5)
            plt.plot(x[m[1]:], y[m[1]:], 'k', lw=0.5)
        plt.show()

        fig, ax = plt.subplots(len(motif.match_indexes), sharex=True, sharey=True)
        for i, (k, m) in enumerate(motif.match_indexes.items()):
            # m[0], m[1] = m[0] * w, m[1] * w
            ax[i].plot(data[k], 'k')
            ax[i].set_ylim((-3, 3))
            ax[i].add_patch(Rectangle((m[0], YMIN), m[1]-m[0], YMAX-YMIN, alpha=0.3))
        plt.subplots_adjust(hspace=0)
        plt.show()


def ostinato(data, m):
    radius, idx, subidx = stumpy.ostinato(data, m)
    consensus_motif = data[idx][subidx : subidx+m]

    # Plot consencus motif and best match in every ts
    nn_idx = []
    for i, ts in enumerate(data):
        if i == idx:
            c, lw = 'b', 1
        else:
            c, lw = 'k', 0.1
        nn_idx.append(np.argmin(stumpy.core.mass(consensus_motif, ts)))
        plt.plot(stumpy.core.z_norm(ts[nn_idx[i] : nn_idx[i]+m]), c, lw=lw)
    plt.show()

    # Plot motif occurences in time series
    fig, ax = plt.subplots(len(data), sharex=True, sharey=True)
    for i, ts in enumerate(data):
        ax[i].plot(ts, 'k')
        ax[i].set_ylim((-3, 3))
        ax[i].add_patch(Rectangle((nn_idx[i], YMIN), m, YMAX-YMIN, alpha=0.3))
    plt.subplots_adjust(hspace=0)
    plt.show()
