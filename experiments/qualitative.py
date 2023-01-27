"""
This module finds patterns in the outlines of images from the MPEG-7 data set.
The data set is freely available from https://dabi.temple.edu/external/shape/MPEG7/dataset.html

After mining motifs from the contours, they are mapped back to the images.
"""
from os import listdir
from os.path import join
import itertools
import re

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from motifminer.miner import Miner

IMG_DIR = 'mpeg7'
NAME = 'fly'

MIN_SUP = 0.5
W = 5
A = 5
L = 5
O = 0.7
K = 10


def main():
    # Load data
    files = [join(IMG_DIR, file) for file in listdir(IMG_DIR) if re.match(f'^{NAME}', file)]
    ts, contours = get_all_ts(files)

    # Mine motifs
    miner = Miner(ts, MIN_SUP, W, A, L, O, K, False)
    motifs = miner.mine_motifs()

    # Plot motifs
    plot_contours(motifs, contours, W)
    plot_ts(motifs, contours, ts, W)


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

    ts_all = tf.ragged.constant(ts_all, dtype=tf.float32)
    return ts_all, contour_all


def get_contour(im):
    """Get the contour of an image.

    Given a single image file, extracts the longest contour using opencv.
    Then, calculates the center of the contour.
    """
    a = np.asarray(im)
    contours, hierarchy = cv2.findContours(a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=lambda c: c.shape[0], reverse=True)

    # Center first contour
    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY), [(point[0][0], point[0][1]) for point in contours[0]]


def get_ts(center, contour):
    """Compute distances from contour points to center.

    Uses the radial distance method (see https://izbicki.me/blog/converting-images-into-time-series-for-data-mining).
    For each point in the contour, the radial distance is calculated.
    """
    ts = []
    for point in contour:
        d = ((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) ** 0.5
        ts.append(d)
    ts = np.array(ts)
    ts = (ts - np.mean(ts)) / np.std(ts)  # z-norm
    return ts


def plot_ts(motifs, contours, ts, w):
    """Plot the time series with motifs that occur in them."""
    for i, (contour, t) in enumerate(zip(contours, ts)):
        if i > 5:
            continue

        fig, axd = plt.subplot_mosaic([['ts', 'contour']], figsize=(15, 3), gridspec_kw={'width_ratios': [4, 1]})
        fig.set_dpi(300)

        # Plot motifs in ts
        axd['ts'].plot(t, 'k', lw=0.5)
        axd['ts'].set(xticks=[0, len(t)], yticks=[])
        plot_motifs(motifs, i, list(range(len(t))), t, w, axd['ts'])

        # Plot motifs in corresponding figure
        plot_contour(motifs, contour, i, w, axd['contour'])
        plt.show()


def plot_contours(motifs, contours, w):
    """Plot the images with motifs that occur in them."""
    fig, axs = plt.subplots(ncols=5, nrows=4)
    fig.set_dpi(300)
    for (i, contour), ax in zip(enumerate(contours), itertools.chain.from_iterable(axs)):
        plot_contour(motifs, contour, i, w, ax)

    plt.show()


def plot_contour(motifs, contour, i, w, ax):
    x, y = contour_to_xy(contour)
    ax.set(xticks=[], yticks=[])
    ax.plot(x, y, 'k', lw=0.5)
    plot_motifs(motifs, i, x, y, w, ax)


def contour_to_xy(contour):
    """Convert contour to a list of (x,y) points."""
    x, y = list(zip(*contour))

    max_x = max(x)
    max_y = max(y)

    return [i / max_x for i in x], [i / max_y for i in y]


def plot_motifs(motifs, i, x, y, w, ax):
    """Plot all motifs occurring in a unit."""
    colors = [
        'maroon', 'steelblue', 'olive', 'salmon', 'teal', 'seagreen', 'purple', 'goldenrod', 'orange', 'tomato'
    ]
    for motif, color in zip(motifs, colors):
        if m := motif.match_indexes.get(i, False):
            start, end = m[0] * w, m[1] * w + 1
            ax.plot(x[start : end], y[start : end], color, lw=1.5)


if __name__ == '__main__':
    main()
