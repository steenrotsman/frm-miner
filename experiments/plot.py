"""Plot utility.

Defines common functions for plotting results from the experiments
"""
import matplotlib.pyplot as plt

from frm.preprocessing import breakpoints


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


if __name__ == '__main__':
    import numpy as np
    from numpy.random import randint as ri

    from frm.miner import Miner
    from frm.preprocessing import sax

    LENGTH = 256
    np.random.seed(42)
    steps = np.linspace(0, np.pi, LENGTH)

    D = np.array([np.sum([np.sin(ri(1, 10) * steps) for _ in range(ri(1, 10))], axis=0) for _ in range(5)])
    mm = Miner(D, 0.6, 16, 4, 3, 0.9)
    mm.mine_motifs()

    colors = ['maroon', 'steelblue', 'olive', 'salmon', 'teal', 'seagreen', 'purple', 'goldenrod', 'orange', 'tomato']

    fig, axs = plt.subplots(ncols=5, figsize=(16, 4), layout='compressed')
    fig.set_dpi(300)
    fig.align_labels()
    # Time series database
    axs[0].plot(D.T)
    axs[0].set(ylim=(-4, 4), xticks=[0, LENGTH], yticks=[], xlabel='Time series database')
    remove_spines(axs[0])

    # Sequence database
    for s, y in zip(mm.sequences, range(7, -1, -1)):
        axs[1].text(0, y/10, s, fontsize=22)
    axs[1].set(xticks=[], yticks=[], xlabel='Sequence database')
    remove_spines(axs[1])

    # Sequence motifs
    for motif, y in zip(mm.motifs, range(7, -1, -1)):
        axs[2].text(0.3, y/10+0.025, motif.pattern, fontsize=22)
    axs[2].set(xticks=[], yticks=[], xlabel='Sequence motifs')
    remove_spines(axs[2])

    # Occurrences and representative motif
    for i, ts in enumerate(D):
        axs[3].plot(ts, 'k', lw=0.5)
        for motif, color in zip(mm.motifs, colors):
            if i in motif.match_indexes:
                start, end = motif.match_indexes[i]
                axs[3].plot(list(range(start, end)), ts[start:end], color, lw=1)

            axs[4].plot(motif.representative, color)

    axs[3].set(xticks=[0, LENGTH], yticks=[], xlabel='Occurrences')
    axs[4].set(xticks=[0, 64], yticks=[], xlabel='Representative motifs')
    remove_spines(axs[3])
    remove_spines(axs[4])

    plt.show()

