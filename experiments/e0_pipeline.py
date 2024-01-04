"""
This module finds patterns in the outlines of images from the MPEG-7 data set.
The data set is freely available from https://dabi.temple.edu/external/shape/MPEG7/dataset.html
"""
from os.path import join
from itertools import chain

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from frm._frm_py.miner import Miner
from frm._frm_py.preprocessing import standardise, sax, get_breakpoints
from plot import remove_spines, COLORS, WIDTH

IMG_DIR = 'mpeg7'
NAME = 'cattle'
SAMPLE = True
ALPHABET = 5
LENGTH = 256


def main():
    # Load data
    files = [join(IMG_DIR, f'{NAME}-{i + 1}.gif') for i in range(3)]
    ts, contours = get_all_ts(files)

    plot = Pipeline(2/3, 16, ALPHABET, LENGTH)
    fig, axs = plt.subplots(ncols=5, figsize=(WIDTH*2, WIDTH/2), layout='compressed')
    plot.plot(fig, axs, ts, 'pipeline', [plot.timeseries, plot.sequences, plot.sequence_motifs, plot.occurrences, plot.representative_motifs])

    plot = Pipeline(2/3, 32, ALPHABET, LENGTH)
    fig, axs = plt.subplots(ncols=3, figsize=(WIDTH*2, WIDTH/2), layout='compressed')
    plot.plot(fig, axs, [ts[0]], 'sax', [plot.timeseries, plot.sax, plot.sequences])

    plot = Pipeline(2/3, 16, ALPHABET, LENGTH)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(WIDTH*2, WIDTH), layout='compressed')
    plot.plot(fig, chain.from_iterable(axs), ts, 'long', [plot.timeseries, plot.sax, plot.sequences, plot.sequence_motifs, plot.occurrences, plot.representative_motifs])


class Pipeline:
    def __init__(self, minsup, seglen, alphabet, length):
        self.minsup = minsup
        self.seglen = seglen
        self.alphabet = alphabet

        self.length = length

        self.mm = Miner(minsup, seglen, alphabet)

        self.data = None

    def plot(self, fig, axs, data, name, steps):
        fig.align_labels()

        self.data = standardise([row[:self.length] for row in data])
        self.mm.mine(self.data)

        for ax, step in zip(axs, steps):
            step(ax)
            remove_spines(ax)

        plt.savefig(join('figs', f'0 {name}.eps'))
        plt.savefig(join('figs', f'0 {name}.png'))
        plt.close()

    def timeseries(self, ax):
        ax.plot(self.data.T, lw=0.5)
        ax.set(ylim=(-3, 3), xticks=[0, self.length-1], yticks=[], xlabel='Time series database')

    def sax(self, ax):
        for b in get_breakpoints(self.alphabet):
            ax.axhline(y=b, color='lightgrey', lw=0.5)

        ax.plot(self.data[0], lw=0.5)
        sequence = sax([self.data[0]], self.seglen, self.alphabet)[0]
        for i in range(0, len(self.data[0]), self.seglen):
            start_idx = i
            end_idx = i + self.seglen
            x_values = np.arange(start_idx, end_idx)
            y_value = np.mean(self.data[0][start_idx:end_idx])
            ax.plot(x_values, np.full_like(x_values, y_value, dtype=y_value.dtype), color='black', lw=0.5)

            middle_idx = (start_idx + end_idx) // 2
            ax.text(middle_idx, y_value - 0.25, sequence[i // self.seglen], size='small', ha='center', va='center', color='black')

        ax.set(ylim=(-3, 3), xticks=[0, self.length-1], yticks=[], xlabel='SAX')

    def sequences(self, ax):
        for s, y in zip(sax(self.data, self.seglen, self.alphabet), range(7, -1, -1)):
            ax.text(0.5, y / 10, s, fontsize=6, ha='center', va='center')
        ax.set(xticks=[], yticks=[], xlabel='Sequence database')

    def sequence_motifs(self, ax):
        for motif, y, color in zip(self.mm.motifs, range(7, -1, -1), COLORS):
            ax.text(0.3, y / 10 + 0.025, motif.pattern, fontsize=6, color=color)
        ax.set(xticks=[], yticks=[], xlabel='Sequence motifs')

    def occurrences(self, ax):
        for i, ts in enumerate(self.data):
            ax.plot(ts, 'k', lw=0.25)
            for motif, color in zip(self.mm.motifs, COLORS):
                if i in motif.best_matches:
                    start = motif.best_matches[i]
                    end = start + motif.length
                    ax.plot(list(range(start, end)), ts[start:end], color, lw=0.5)

        ax.set(ylim=(-3, 3), xticks=[0, self.length-1], yticks=[], xlabel='Occurrences')

    def representative_motifs(self, ax):
        for motif, color in zip(self.mm.motifs, COLORS):
            ax.plot(motif.representative, color, lw=0.5)

        ax.set(ylim=(-3, 3), xticks=[0, self.length-1], yticks=[], xlabel='Representative motifs')

    def empty(self, ax):
        ax.text(0.5, 0.5, '...')
        ax.set(xticks=[], yticks=[])


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


def sample(ts, sample=512, cutoff=256):
    step = len(ts) / sample
    res = []
    # Bresenham interpolation
    a = 0
    while a < len(ts):
        if len(res) == sample:
            break
        res.append(ts[int(a)])
        a = a + step
    return res[:cutoff]


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
