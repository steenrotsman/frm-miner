"""
This module finds patterns in the outlines of images from the MPEG-7 data set.
The data set is freely available from https://dabi.temple.edu/external/shape/MPEG7/dataset.html
"""

from itertools import chain
from os.path import join

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import text, transforms
from PIL import Image
from plot import HEIGHT, WIDTH, colors, remove_spines

from frm import Miner
from frm.patterns import PatternMiner
from frm.preprocessing import get_breakpoints, sax, standardise

IMG_DIR = "mpeg7"
NAME = "cattle"
SAMPLE = True
ALPHA = 5
LENGTH = 256


def main():
    # Load data
    files = [join(IMG_DIR, f"{NAME}-{i + 1}.gif") for i in range(3)]
    ts, contours = get_all_ts(files)

    # # Basic pipelines
    # plot = Pipeline(2 / 3, 16, ALPHA, LENGTH)
    # fig, axs = plt.subplots(ncols=5, figsize=(WIDTH, HEIGHT))
    # steps = [plot.D, plot.Ds, plot.sm, plot.occ, plot.rm]
    # plot.plot(fig, axs, ts, 'pipeline', steps)

    # plot = Pipeline(2 / 3, 32, ALPHA, LENGTH)
    # fig, axs = plt.subplots(ncols=3, figsize=(WIDTH, HEIGHT))
    # steps = [plot.D, plot.sax, plot.Ds]
    # plot.plot(fig, axs, [ts[0]], 'sax', steps)

    # plot = Pipeline(2 / 3, 16, ALPHA, LENGTH)
    # fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(WIDTH, HEIGHT * 2))
    # steps = [plot.D, plot.sax, plot.Ds, plot.sm, plot.occ, plot.rm]
    # plot.plot(fig, chain.from_iterable(axs), ts, 'long', steps)

    # Large pipeline
    large_pipe(ts)


def large_pipe(ts):
    fig = plt.figure(figsize=(WIDTH, WIDTH * 1.2), layout="none")
    axd = {}
    outer_grid = fig.add_gridspec(3, 3, wspace=0.1, hspace=0.5)
    inner_grids = []
    inner_grids.append(outer_grid[0, 0].subgridspec(3, 1, hspace=0))
    inner_grids.append(outer_grid[0, 1].subgridspec(3, 1, hspace=0))
    inner_grids.append(outer_grid[0, 2].subgridspec(1, 1))
    inner_grids.append(outer_grid[1, 0].subgridspec(1, 1))
    inner_grids.append(outer_grid[1, 1].subgridspec(1, 1))
    inner_grids.append(outer_grid[1, 2].subgridspec(1, 1))
    inner_grids.append(outer_grid[2, 0].subgridspec(3, 1, hspace=0))
    inner_grids.append(outer_grid[2, 1].subgridspec(3, 1, hspace=0))
    inner_grids.append(outer_grid[2, 2].subgridspec(1, 1))

    keys = ["ts", "sax", "seq", "sms", "red", "seq_occ", "ts_occ", "ao", "rm"]
    for gs, key in zip(inner_grids, keys):
        axd[key] = gs.subplots()

    minsup = 2 / 3
    alpha = ALPHA
    seglen = 32

    # Time series
    data = standardise([row[:256] for row in ts])
    for row, ax in zip(data, axd["ts"]):
        ax.plot(row)
        ax.set(xticks=[])
        remove_spines(ax)
    axd["ts"][-1].set(xticks=[0, len(row) - 1], xlabel="Time series")

    # SAX
    for row, ax in zip(data, axd["sax"]):
        for b in get_breakpoints(ALPHA):
            ax.axhline(y=b, color="lightgrey", lw=0.5)

        ax.plot(row)
        sequence = sax([row], seglen, alpha)[0]
        for i in range(0, len(row), seglen):
            start_idx = i
            end_idx = i + seglen
            x_values = np.arange(start_idx, end_idx)
            y_value = np.mean(row[start_idx:end_idx])
            y_values = np.full_like(x_values, y_value, dtype=y_value.dtype)
            ax.plot(x_values, y_values, color="black", lw=0.5)

            middle_idx = (start_idx + end_idx) // 2
            ax.text(
                middle_idx,
                y_value + 0.7,
                sequence[i // seglen],
                ha="center",
                va="center",
            )
            ax.set(xticks=[])
            remove_spines(ax)
    axd["sax"][-1].set(xticks=[0, len(row) - 1], xlabel="SAX")

    # Sequence database
    sequences = sax(data, seglen, alpha)
    for sequence, y in zip(sequences, [0.7, 0.6, 0.5]):
        axd["seq"].text(0.5, y, sequence, ha="center", va="center")
    axd["seq"].set(xticks=[])  #  , xlabel='Sequence database')
    label = text.Text(0.5, -0.18, "Sequence database", ha="center", va="center")
    label.set_transform(
        transforms.offset_copy(
            axd["seq"].transAxes, axd["seq"].figure, x=5, y=0, units="points"
        )
    )
    label.set_clip_on(False)
    axd["seq"].add_artist(label)
    remove_spines(axd["seq"])

    # Sequence motifs
    pm = PatternMiner(minsup, omax=1)
    pm.mine(sax(data, seglen, alpha))
    coords = [(x, y) for x in (0.2, 0.45, 0.75) for y in range(9, 0, -1)]
    for pattern, (x, y) in zip(pm.frequent, coords):
        axd["sms"].text(x, y / 10, pattern, ha="center", va="center")
    axd["sms"].set(xticks=[], xlabel="Sequence motifs")
    remove_spines(axd["sms"])

    # Redundancy elimination
    pm.omax = 0.8
    pm.remove_redundant()
    for pattern, y, c in zip(pm.frequent, (0.6, 0.4), ("k", colors[1])):
        axd["red"].text(0.5, y, pattern, ha="center", va="center", color=c)
    axd["red"].set(xticks=[], xlabel="Redundancy elimination")
    remove_spines(axd["red"])

    # Sequence motif occurrences
    motif = list(pm.frequent.values())[1]
    for (i, sequence), y in zip(enumerate(sequences), [0.6, 0.5, 0.4]):
        for j, char in enumerate(sequence):
            color = (
                colors[1]
                if i in motif.indexes
                and j
                in range(motif.indexes[i][0], motif.indexes[i][0] + len(motif.pattern))
                else "black"
            )
            axd["seq_occ"].text(
                0.35 + j * 0.04, y, char, ha="center", va="center", color=color
            )
    axd["seq_occ"].set(xticks=[], yticks=[], xlabel="Sequence occurrences")
    remove_spines(axd["seq_occ"])

    # Time series occurrences
    for i, (row, ax) in enumerate(zip(data, axd["ts_occ"])):
        if i in motif.indexes:
            start = motif.indexes[i][0] * seglen
            end = start + len(motif.pattern) * seglen
            ax.plot(range(start), row[:start], color=colors[0])
            ax.plot(range(start, end), row[start:end], color=colors[1])
            ax.plot(range(end, len(row)), row[end:], color=colors[0])
        else:
            ax.plot(row)
        ax.set(xticks=[])
        remove_spines(ax)
    axd["ts_occ"][-1].set(xticks=[0, len(row) - 1], xlabel="Time series occurrences")

    # Average occurrences
    for i, (row, ax) in enumerate(zip(data, axd["ao"])):
        if i in motif.indexes:
            start = motif.indexes[i][0] * seglen
            end = start + len(motif.pattern) * seglen
            ax.plot(row[start:end], color=colors[1])
        else:
            ax.text(0.5, 0.5, "No occurrences", ha="center", va="center")
        ax.set(xticks=[])
        remove_spines(ax)
    axd["ao"][-1].set(
        xticks=[0, len(motif.pattern) * seglen - 1], xlabel="Average occurrences"
    )

    # Representative motif
    occurrences = []
    for i, row in enumerate(data):
        if i in motif.indexes:
            start = motif.indexes[i][0] * seglen
            end = start + len(motif.pattern) * seglen
            occurrences.append(row[start:end])
    rm = np.mean(occurrences, axis=0)
    axd["rm"].plot(rm, color=colors[1])
    axd["rm"].set(
        xticks=[0, len(motif.pattern) * seglen - 1],
        xlabel="Representative motif",
        ylim=[-1.7, 11],
    )
    remove_spines(axd["rm"])

    plt.subplots_adjust(top=0.98, bottom=0.07, left=0.01, right=0.99)
    plt.savefig("figs/pipe.eps")
    plt.savefig("figs/pipe.png")


class Pipeline:
    def __init__(self, minsup, seglen, alpha, length):
        self.minsup = minsup
        self.seglen = seglen
        self.alpha = alpha

        self.length = length

        self.mm = Miner(minsup, seglen, alpha)

        self.data = None

        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def plot(self, fig, axs, data, name, steps):
        fig.align_labels()

        self.data = standardise([row[: self.length] for row in data])
        self.mm.mine(self.data)

        for ax, step in zip(axs, steps):
            step(ax)
            remove_spines(ax)

        plt.savefig(join("figs", f"0 {name}.eps"))
        plt.savefig(join("figs", f"0 {name}.png"))
        plt.close()

    def D(self, ax):
        ax.plot(self.data.T, lw=0.5)
        self.axset(ax, "Time series")

    def sax(self, ax):
        for b in get_breakpoints(self.alpha):
            ax.axhline(y=b, color="lightgrey", lw=0.5)

        ax.plot(self.data[0], lw=0.5)
        sequence = sax([self.data[0]], self.seglen, self.alpha)[0]
        for i in range(0, len(self.data[0]), self.seglen):
            start_idx = i
            end_idx = i + self.seglen
            x_values = np.arange(start_idx, end_idx)
            y_value = np.mean(self.data[0][start_idx:end_idx])
            y_values = np.full_like(x_values, y_value, dtype=y_value.dtype)
            ax.plot(x_values, y_values, color="black", lw=0.5)

            middle_idx = (start_idx + end_idx) // 2
            ax.text(
                middle_idx,
                y_value + 0.5,
                sequence[i // self.seglen],
                ha="center",
                va="center",
                color="black",
            )

        self.axset(ax, "SAX")

    def Ds(self, ax):
        sequences = "\n".join(sax(self.data, self.seglen, self.alpha))
        ax.text(0.5, 0.5, sequences, ha="center", va="center")
        ax.set(xticks=[], yticks=[], xlabel="Sequence")

    def sm(self, ax):
        for motif, y, color in zip(self.mm.motifs, range(4, -1, -1), self.colors):
            ax.text(0.5, y / 5, motif.pattern, ha="center", c=color)
        ax.set(xticks=[], yticks=[], xlabel="Sequence motifs")

    def occ(self, ax):
        for i, ts in enumerate(self.data):
            ax.plot(ts, "k", lw=0.25)
            for motif, color in zip(self.mm.motifs, self.colors):
                if i in motif.best_matches:
                    start = motif.best_matches[i]
                    end = start + motif.length
                    ax.plot(list(range(start, end)), ts[start:end], c=color, lw=0.5)

        self.axset(ax, "Occurrences")

    def rm(self, ax):
        for motif in self.mm.motifs:
            ax.plot(motif.representative, lw=0.5)

        self.axset(ax, "Representative motifs")

    def empty(self, ax):
        ax.text(0.5, 0.5, "...")
        ax.set(xticks=[], yticks=[])

    def axset(self, ax, xlabel):
        ax.set(ylim=(-3, 3), xticks=[0, self.length - 1], yticks=[], xlabel=xlabel)


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
    ax.plot(x, y, "k", lw=0.5)
    plot_motifs(motifs, i, x, y, ax)


def contour_to_xy(contour):
    """Convert contour to a list of (x,y) points."""
    x, y = list(zip(*contour))

    max_x = max(x)
    max_y = max(y)

    return [i / max_x for i in x], [i / max_y for i in y]


def plot_motifs(motifs, i, x, y, ax):
    """Plot all motifs occurring in a unit."""
    for motif in zip(motifs):
        if m := motif.best_matches.get(i, False):
            start = m
            end = start + motif.length
            ax.plot(x[start:end], y[start:end], lw=1.5)


if __name__ == "__main__":
    main()
