import csv
from collections import namedtuple
from itertools import product
from multiprocessing import Pool
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from plot import plot_motifs

from frm import Miner

# Simulation settings
UNITS = 10000
TS_LEN = 10000
SEED = 1234
NOISE = 2
OVERLAP = 0.5
RUNS = 1000
PATH = "e2_problem_setting.csv"
Setting = namedtuple("Setting", ["name", "support", "motif"])

LENGTHS = [200, 250, 300, 350]
SETTINGS = [
    Setting("up", 9000, np.linspace(-3, 3, LENGTHS[0]) ** 2),
    Setting("peak", 8500, np.linspace(4, -4, LENGTHS[1])),
    Setting("down", 8000, -(np.linspace(-3, 3, LENGTHS[2]) ** 2)),
    Setting("dip", 7500, np.linspace(-4, 4, LENGTHS[3])),
]
FIELDS = ["run"] + [f"{s.name}{x}" for s in SETTINGS for x in ["", "_r", "_l"]]


# Parameters
MINSUP = 0.3
SEGLEN = 30
ALPHA = 4
OMAX = 0.6
DIFF = 1
K = 4


def main():
    with open(PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
    with Pool(8) as p:
        list(p.imap_unordered(single_run, range(RUNS)))


def single_run(run):
    ts, ground_truth = get_data()
    motifs = get_motifs(ts)
    if run == 0:
        plot_data_and_results(ts, ground_truth, motifs)
    result = evaluate_motifs(ground_truth, motifs)
    with open(PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow({"run": run, **result})


def get_data(run=0):
    # Make each run independently reproducible
    rng = np.random.default_rng(SEED + run)

    # Starting indices for motifs in each time series
    locations = [
        rng.integers(0, 1500, size=UNITS),
        rng.integers(2500, 4000, size=UNITS),
        rng.integers(5000, 6500, size=UNITS),
        rng.integers(7500, 9000, size=UNITS),
    ]

    # Gaussian noise
    ts = rng.normal(0, NOISE, size=(UNITS, TS_LEN))

    # Insert motifs
    ground_truth = []
    for locations, length, setting in zip(locations, LENGTHS, SETTINGS):
        # Only keep starting indices for motif-specific support number of time series
        locations[setting.support :] = -1
        rng.shuffle(locations)
        ground_truth.append(locations)
        for i, loc in enumerate(locations):
            if loc == -1:
                continue
            noisy_motif = setting.motif + rng.normal(size=length)
            ts[i][loc : loc + length] = noisy_motif
    return np.cumsum(ts, axis=1), ground_truth


def get_motifs(ts):
    mm = Miner(MINSUP, SEGLEN, ALPHA, k=K, diff=DIFF, omax=OMAX)
    motifs = mm.mine(ts)
    return motifs


def plot_data_and_results(ts, ground_truth, motifs):
    fig, ax = plt.subplots(nrows=1, sharex=True)
    # Select rows with and without motifs
    for row, indices in zip(ts, ground_truth):
        ts_index = np.where(indices == -1)[0][0]
        ax.plot(ts[ts_index])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(join("figs", "Fig3.pdf"))

    fig, axs = plt.subplots(ncols=K, sharey="all")
    plot_motifs(axs, ts, motifs, fn="Fig4")


def evaluate_motifs(ground_truth, motifs):
    # Convert starting indices to (start, end) ranges for ground truth motifs
    gt_intervals = [
        {
            ts_idx: (start, start + length)
            for ts_idx, start in enumerate(locations)
            if start != -1
        }
        for locations, length in zip(ground_truth, LENGTHS)
    ]

    # Initialize result
    result = {f"{s.name}{x}": 0 for s in SETTINGS for x in ["", "_r", "_l"]}

    # Check if the discovered motifs correspond to the embedded motifs
    for (gt, setting), motif in product(zip(gt_intervals, SETTINGS), motifs):
        if match(gt, motif):
            result[setting.name] = 1
            result[f"{setting.name}_r"] = len(motif.best_matches) / setting.support
            result[f"{setting.name}_l"] = motif.length / len(setting.motif)
    return result


def match(gt, motif):
    overlapping = sum(
        1
        for ts_idx, start in motif.best_matches.items()
        if ts_idx in gt and cover(start, start + motif.length, *gt[ts_idx]) > OVERLAP
    )
    return overlapping >= len(motif.best_matches) / 2


def cover(start, end, gt_start, gt_end):
    overlap = max(0, min(end, gt_end) - max(start, gt_start))
    return overlap / (gt_end - gt_start)


if __name__ == "__main__":
    main()
