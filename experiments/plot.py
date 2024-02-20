from os.path import join

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

WIDTH = 0.0138 * 347.12354 / 2
HEIGHT = WIDTH / 2
colors = cycler(color=['blue', 'darkkhaki', 'chocolate', 'mediumseagreen'])
params = {
    'axes.labelsize': 6,
    'axes.prop_cycle': colors,
    'figure.constrained_layout.use': True,
    'figure.figsize': [WIDTH, HEIGHT],
    'font.family': 'serif',
    'font.size': 6,
    'legend.fontsize': 6,
    'savefig.dpi': 2400,
    'text.usetex': False,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
}
matplotlib.rcParams.update(params)


def plot_motifs(flat_axs, data, motifs, length=0, fn='', **axset):
    xticks = 'xticks' in axset
    for motif, ax in zip(motifs, flat_axs):
        # If length is not supplied, use length of the motif
        limit = len(motif.representative) if not length else length

        # Plot matches of the representative motif
        for seq, index in motif.best_matches.items():
            ax.plot(data[seq][index : index + motif.length], 'k', lw=0.1)

        # Plot representative motif
        ax.plot(motif.representative, 'b', lw=1)

        # Aesthetics
        if not xticks:
            axset['xticks'] = [0, limit]
        ax.set(**axset)
        remove_spines(ax, remove_y=False)
    if fn:
        plt.savefig(join('figs', f'{fn}.eps'))
        plt.savefig(join('figs', f'{fn}.png'))
        plt.close()


def remove_spines(ax, remove_y=True):
    if remove_y:
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


if __name__ == '__main__':
    from PIL import Image

    for i in range(4):
        plt.plot([i, i])
    plt.savefig(join('figs', 'colors.png'))
    img = Image.open(join('figs', 'colors.png'))
    img_gray = img.convert('L')
    img_gray.save(join('figs', 'grayscale.png'))
