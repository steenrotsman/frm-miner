from os.path import join

import matplotlib
import matplotlib.pyplot as plt

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
    'savefig.dpi': 2400
}
matplotlib.rcParams.update(params)

COLORS = ['maroon', 'steelblue', 'olive', 'salmon', 'teal', 'seagreen', 'purple', 'goldenrod', 'orange', 'tomato']


def plot_motifs(flat_axs, data, motifs, length=0, fn=''):
    for motif, ax in zip(motifs, flat_axs):
        # If length is not supplied, use length of the motif
        limit = len(motif.representative) if not length else length

        # Plot matches of the representative motif
        for seq, index in motif.best_matches.items():
            ax.plot(data[seq][index : index+motif.length], 'k', lw=0.1)

        # Plot representative motif
        ax.plot(motif.representative, 'b', lw=1)

        # Aesthetics
        ax.set(xticks=[0, limit])
        remove_spines(ax, remove_y=False)
    if fn:
        plt.savefig(join('figs', f'{fn}.eps'))
        plt.close()


def remove_spines(ax, remove_y=True):
    if remove_y:
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
