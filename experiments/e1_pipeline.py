"""
This module finds patterns in the outlines of images from the MPEG-7 data set.
The data set is freely available from https://dabi.temple.edu/external/shape/MPEG7/dataset.html
"""

from os.path import join

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import text, transforms
from PIL import Image
from plot import WIDTH, colors, remove_spines

from frm.patterns import PatternMiner
from frm.preprocessing import get_breakpoints, sax, standardise

IMG_DIR = "mpeg7"
NAME = "cattle"
SAMPLE = True
MINSUP = 2 / 3
ALPHA = 5
SEGLEN = 32
LENGTH = 256


def main():
    # Load data
    files = [join(IMG_DIR, f"{NAME}-{i + 1}.gif") for i in range(3)]
    ts, contours = get_all_ts(files)
    small_pipe(ts)
    large_pipe(ts)


def small_pipe(ts):
    fig, axs = plt.subplots(ncols=3)

    # Time series
    axs[0].plot(ts[0], lw=0.5)
    axs[0].set(ylim=(-3, 3), xticks=[0, len(ts) - 1], yticks=[], xlabel="Time series")

    # SAX breakpoints
    for b in get_breakpoints(ALPHA):
        axs[1].axhline(y=b, color="lightgrey", lw=0.5)

    # SAX segments
    axs[1].plot(ts[0], lw=0.5)
    sequence = sax([ts[0]], SEGLEN, ALPHA, 0)[0]
    for i in range(0, len(ts[0]), SEGLEN):
        start_idx = i
        end_idx = i + SEGLEN
        x_values = np.arange(start_idx, end_idx)
        y_value = np.mean(ts[0][start_idx:end_idx])
        y_values = np.full_like(x_values, y_value, dtype=y_value.dtype)
        axs[1].plot(x_values, y_values, color="black", lw=0.5)

        middle_idx = (start_idx + end_idx) // 2
        axs[1].text(
            middle_idx,
            y_value + 0.5,
            sequence[i // SEGLEN],
            ha="center",
            va="center",
            color="black",
        )

    axs[1].set(ylim=(-3, 3), xticks=[0, len(ts) - 1], yticks=[], xlabel="SAX")

    # Sequences
    sequences = "\n".join(sax(ts, SEGLEN, ALPHA, 0))
    axs[2].text(0.5, 0.5, sequences, ha="center", va="center")
    axs[2].set(xticks=[], yticks=[], xlabel="Sequence")


def large_pipe(ts):
    # Set up nested grid (3x3 cells where some cells contain 3 rows)
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
        sequence = sax([row], SEGLEN, ALPHA, 0)[0]
        for i in range(0, len(row), SEGLEN):
            start_idx = i
            end_idx = i + SEGLEN
            x_values = np.arange(start_idx, end_idx)
            y_value = np.mean(row[start_idx:end_idx])
            y_values = np.full_like(x_values, y_value, dtype=y_value.dtype)
            ax.plot(x_values, y_values, color="black", lw=0.5)

            middle_idx = (start_idx + end_idx) // 2
            ax.text(
                middle_idx,
                y_value + 0.7,
                sequence[i // SEGLEN],
                ha="center",
                va="center",
            )
            ax.set(xticks=[])
            remove_spines(ax)
    axd["sax"][-1].set(xticks=[0, len(row) - 1], xlabel="SAX")

    # Sequence database
    sequences = sax(data, SEGLEN, ALPHA, 0)
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
    pm = PatternMiner(MINSUP, omax=1)
    pm.mine(sax(data, SEGLEN, ALPHA, 0))
    coords = [(x, y) for x in (0.2, 0.4, 0.6, 0.8) for y in range(8, 1, -1)]
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
            start = motif.indexes[i][0] * SEGLEN
            end = start + len(motif.pattern) * SEGLEN
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
            start = motif.indexes[i][0] * SEGLEN
            end = start + len(motif.pattern) * SEGLEN
            ax.plot(row[start:end], color=colors[1])
        else:
            ax.text(0.5, 0.5, "No occurrences", ha="center", va="center")
        ax.set(xticks=[])
        remove_spines(ax)
    axd["ao"][-1].set(
        xticks=[0, len(motif.pattern) * SEGLEN - 1], xlabel="Average occurrences"
    )

    # Representative motif
    occurrences = []
    for i, row in enumerate(data):
        if i in motif.indexes:
            start = motif.indexes[i][0] * SEGLEN
            end = start + len(motif.pattern) * SEGLEN
            occurrences.append(row[start:end])
    rm = np.mean(occurrences, axis=0)
    axd["rm"].plot(rm, color=colors[1])
    axd["rm"].set(
        xticks=[0, len(motif.pattern) * SEGLEN - 1],
        xlabel="Representative motif",
        ylim=[-1.7, 5],
    )
    remove_spines(axd["rm"])

    plt.subplots_adjust(top=0.98, bottom=0.07, left=0.01, right=0.99)
    plt.savefig("figs/Fig1.pdf")


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


if __name__ == "__main__":
    main()
