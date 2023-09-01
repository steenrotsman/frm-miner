"""
This module finds patterns in the outlines of images from the MPEG-7 data set.
The data set is freely available from https://dabi.temple.edu/external/shape/MPEG7/dataset.html
"""
from os.path import join

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from frm._frm_py.miner import Miner
from frm._frm_py.preprocessing import standardise, sax
from plot import remove_spines, COLORS, WIDTH

IMG_DIR = 'mpeg7'
NAME = 'cattle'
SAMPLE = True
ALPHABET = 5


def main():
    # Load data
    files = [join(IMG_DIR, f'{NAME}-{i + 1}.gif') for i in range(20)]
    ts, contours = get_all_ts(files)

    # Plot pipeline
    plot_pipeline(ts, 256, 3)


def plot_pipeline(ts, length, n):
    minsup = (n-1) / n
    seglen = int(length / 16)
    alphabet = ALPHABET
    data = standardise([t[:length] for t in ts[:n]])
    mm = Miner(minsup, seglen, alphabet)
    mm.mine(data)

    fig, axs = plt.subplots(ncols=5, figsize=(WIDTH*2, WIDTH/2), layout='compressed')
    fig.align_labels()

    # Time series database
    axs[0].plot(data.T, lw=0.5)
    axs[0].set(ylim=(-3, 3), xticks=[0, length], yticks=[], xlabel='Time series database')
    remove_spines(axs[0])

    # Sequence database
    for s, y in zip(sax(data, seglen, alphabet), range(7, -1, -1)):
        axs[1].text(0, y / 10, s, fontsize=6)
    axs[1].set(xticks=[], yticks=[], xlabel='Sequence database')
    remove_spines(axs[1])

    # Sequence motifs
    for motif, y, color in zip(mm.motifs, range(7, -1, -1), COLORS):
        axs[2].text(0.3, y / 10 + 0.025, motif.pattern, fontsize=6, color=color)
    axs[2].set(xticks=[], yticks=[], xlabel='Sequence motifs')
    remove_spines(axs[2])

    # Occurrences and representative motif
    for i, ts in enumerate(data):
        axs[3].plot(ts, 'k', lw=0.25)
        for motif, color in zip(mm.motifs, COLORS):
            if i in motif.best_matches:
                start = motif.best_matches[i]
                end = start + motif.length
                axs[3].plot(list(range(start, end)), ts[start:end], color, lw=0.5)

            axs[4].plot(motif.representative, color, lw=0.5)

    axs[3].set(ylim=(-3, 3), xticks=[0, length], yticks=[], xlabel='Occurrences')
    axs[4].set(ylim=(-3, 3), xticks=[0, length], yticks=[], xlabel='Representative motifs')
    remove_spines(axs[3])
    remove_spines(axs[4])

    plt.savefig(join('figs', '0 pipeline.eps'))


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
        if m := motif.best_matches.get(i, False):
            start = m
            end = start + motif.length
            ax.plot(x[start : end], y[start : end], color, lw=1.5)


if __name__ == '__main__':
    main()
