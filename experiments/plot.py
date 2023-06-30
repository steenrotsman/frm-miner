"""Plot utility.

Defines common functions for plotting results from the experiments
"""
from os.path import join

import matplotlib
import matplotlib.pyplot as plt

from frm._frm_py.preprocessing import breakpoints

WIDTH = 0.0138 * 347.12354 / 2
HEIGHT = WIDTH / 2
params = {
    'axes.labelsize': 6,
    'font.size': 6,
    'legend.fontsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'text.usetex': False,
    'font.family': 'serif',
    'figure.figsize': [WIDTH, HEIGHT],
    'savefig.dpi': 1200
}
matplotlib.rcParams.update(params)

COLORS = ['maroon', 'steelblue', 'olive', 'salmon', 'teal', 'seagreen', 'purple', 'goldenrod', 'orange', 'tomato']


def plot_motifs(fig, flat_axs, motifs, alphabet, length=0, fn=''):
    for motif, ax in zip(motifs, flat_axs):
        # If length is not supplied, use length of the motif
        limit = len(motif.representative) if not length else length

        # Plot matches of the representative motif
        for match in motif.matches:
            ax.plot(match, 'k', lw=0.1)

        # Plot representative motif
        ax.plot(motif.representative, 'b', lw=1)

        # Plot gridlines corresponding to the SAX breakpoints
        if alphabet:
            ax.hlines(list(breakpoints[alphabet].keys()), 0, limit, 'k', lw=0.3)

        # Aesthetics
        ax.set(ylim=(-3, 3), xticks=[0, limit], yticks=[])
        remove_spines(ax)
    if fn:
        plt.savefig(join('figs', fn))
        plt.close()


def remove_spines(ax):
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
