"""
This module finds patterns in the outlines of images from the MPEG-7 data set.
The data set is freely available from https://dabi.temple.edu/external/shape/MPEG7/dataset.html

After mining motifs from the contours, they are mapped back to the images.
"""
from os.path import join
import itertools

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from frm.miner import Miner
from frm.preprocessing import standardise
from plot import remove_spines, COLORS, WIDTH

IMG_DIR = 'mpeg7'
NAMES = ['bird', 'fly']
SAMPLE = True

MIN_SUP = 0.5
SEGLEN = 4
ALPHABET = 4
MIN_LEN = 5
MAX_OVERLAP = 0.8
LOCAL = True
K = 10


def main():
    ts_bird, contours_bird, motifs_bird = temp(NAMES[0])
    ts_fly, contours_fly, motifs_fly = temp(NAMES[1])

    # Plot motifs
    plot_contours(motifs_fly, contours_fly)
    plot_ts(motifs_fly, contours_fly, ts_fly)

    # Plot pipeline
    plot_pipeline(ts_bird, 256)


def temp(name):
    # Load data
    files = [join(IMG_DIR, f'{name}-{i + 1}.gif') for i in range(20)]
    ts, contours = get_all_ts(files)

    # Mine motifs
    miner = Miner(ts, MIN_SUP, SEGLEN, ALPHABET, MIN_LEN, MAX_OVERLAP, LOCAL, K)
    motifs = miner.mine_motifs()

    return ts, contours, motifs


def get_all_ts(files: list):
    """Get time series and contours from image files.

    For a list of image files, transforms the images into contours and calculates their distances.
    """
    contour_all = []
    ts_all = []
    for file in files:
        im = Image.open(file)
        center, contour = get_contour(im)
        contour_all.append(contour)
        ts = get_ts(center, contour)
        ts_all.append(ts)

    return ts_all, contour_all


def get_contour(im):
    """Get the contour of an image.

    Given a single image file, extracts the longest contour using opencv.
    Then, calculates the center of the contour.
    """
    a = np.asarray(im)
    contours, hierarchy = cv.findContours(a, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=lambda c: c.shape[0], reverse=True)

    # Center first contour
    m = cv.moments(contours[0])
    c_x = int(m["m10"] / m["m00"])
    c_y = int(m["m01"] / m["m00"])

    return (c_x, c_y), [(point[0][0], point[0][1]) for point in contours[0]]


def get_ts(center, contour):
    """Compute distances from contour points to center.

    Uses the radial distance method (see https://izbicki.me/blog/converting-images-into-time-series-for-data-mining).
    For each point in the contour, the radial distance is calculated.
    """
    ts = []
    for point in contour:
        d = ((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) ** 0.5
        ts.append(d)

    if SAMPLE:
        ts = sample(ts)
    return ts


def sample(ts, length=512):
    step = len(ts)/length
    res = []
    # Bresenham interpolation
    a = 0
    while a < len(ts):
        if len(res) == length:
            break
        res.append(ts[int(a)])
        a = a + step
    return res


def plot_ts(motifs, contours, ts):
    """Plot the time series with motifs that occur in them."""
    for i, (contour, t) in enumerate(zip(contours, ts)):
        fig, axd = plt.subplot_mosaic([['ts', 'contour']], gridspec_kw={'width_ratios': [4, 1]}, layout='constrained')
        fig.set_dpi(1200)

        # Plot motifs in ts
        axd['ts'].plot(t, 'k', lw=0.5)
        axd['ts'].set(xticks=[0, len(t)], yticks=[])
        remove_spines(axd['ts'])
        plot_motifs(motifs, i, list(range(len(t))), t, axd['ts'])

        # Plot motifs in corresponding figure
        plot_contour(motifs, contour, i, axd['contour'])
        plt.show()
        return


def plot_contours(motifs, contours):
    """Plot the images with motifs that occur in them."""
    fig, axs = plt.subplots(ncols=5, nrows=4, figsize=(WIDTH, WIDTH/5*4), layout='constrained')
    fig.set_dpi(1200)

    for (i, contour), ax in zip(enumerate(contours), itertools.chain.from_iterable(axs)):
        plot_contour(motifs, contour, i, ax)

    plt.show()


def plot_contour(motifs, contour, i, ax):
    x, y = contour_to_xy(contour)
    ax.set(xticks=[], yticks=[], ylim=[1, 0])
    ax.plot(x, y, 'k', lw=0.5)
    plot_motifs(motifs, i, x, y, ax)


def contour_to_xy(contour):
    """Convert contour to a list of (x,y) points."""
    x, y = list(zip(*contour))

    max_x = max(x)
    max_y = max(y)

    return [i / max_x for i in x], [i / max_y for i in y]


def plot_motifs(motifs, i, x, y, ax):
    """Plot all motifs occurring in a unit."""
    for motif, color in zip(motifs, COLORS):
        if m := motif.match_indexes.get(i, False):
            start, end = m
            ax.plot(x[start : end], y[start : end], color, lw=1.5)


def plot_pipeline(ts, length):
    D = np.array(standardise([t[:length] for t in ts[:5]]))
    mm = Miner(D, 0.6, int(length/16), 4, 3, 0.8)
    mm.mine_motifs()

    fig, axs = plt.subplots(ncols=5, layout='compressed')
    fig.set_dpi(1200)
    fig.align_labels()

    # Time series database
    axs[0].plot(D.T, lw=0.5)
    axs[0].set(ylim=(-4, 4), xticks=[0, length], yticks=[], xlabel='Time series database')
    remove_spines(axs[0])

    # Sequence database
    for s, y in zip(mm.sequences, range(7, -1, -1)):
        axs[1].text(0, y / 10, s, fontsize=6)
    axs[1].set(xticks=[], yticks=[], xlabel='Sequence database')
    remove_spines(axs[1])

    # Sequence motifs
    for motif, y in zip(mm.motifs, range(7, -1, -1)):
        axs[2].text(0.3, y / 10 + 0.025, motif.pattern, fontsize=6)
    axs[2].set(xticks=[], yticks=[], xlabel='Sequence motifs', )
    remove_spines(axs[2])

    # Occurrences and representative motif
    for i, ts in enumerate(D):
        axs[3].plot(ts, 'k', lw=0.25)
        for motif, color in zip(mm.motifs, COLORS):
            if i in motif.match_indexes:
                start, end = motif.match_indexes[i]
                axs[3].plot(list(range(start, end)), ts[start:end], color, lw=0.5)

            axs[4].plot(motif.representative, color, lw=0.5)

    axs[3].set(xticks=[0, length], yticks=[], xlabel='Occurrences')
    axs[4].set(xticks=[0, length/2], yticks=[], xlabel='Representative motifs')
    remove_spines(axs[3])
    remove_spines(axs[4])

    plt.show()


if __name__ == '__main__':
    main()
