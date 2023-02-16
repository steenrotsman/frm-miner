"""Plot utility.

Defines common functions for plotting results from the experiments
"""
import matplotlib.pyplot as plt

from motifminer.preprocessing import breakpoints


def plot_motifs(fig, flat_axs, motifs, alphabet, length=0):
    fig.set_dpi(300)
    fig.tight_layout()

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
            ax.hlines(breakpoints[alphabet].values(), 0, limit, 'k', lw=0.3)

        # Aesthetics
        ax.set(ylim=(-3, 3), xticks=[0, limit], yticks=[])
        remove_spines(ax)
    plt.show()


def remove_spines(ax):
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
