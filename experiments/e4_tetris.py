from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from plot import HEIGHT, WIDTH, plot_motifs

from frm import Miner

COLS = ["gaze_angle_x", "gaze_angle_y", "ear"]
OBS = 23400
EXCLUDE = "B152030042"
DIR = "tetris"

MINSUP = 0.3
SEGLEN = 10
ALPHA = 4
DIFF = 1
OMAX = 0.9
K = 3


def main():
    fig, axs = plt.subplots(ncols=3, figsize=(WIDTH, HEIGHT))

    # Gaze Y data
    ts, subjects = get_data(COLS[1])
    miner = Miner(MINSUP, SEGLEN, ALPHA, diff=DIFF, omax=OMAX, k=K)
    motifs = miner.mine(ts)
    plot_motifs(axs[:2], ts, motifs[:2])

    # EAR data
    ts, subjects = get_data(COLS[2])
    miner = Miner(MINSUP, SEGLEN, ALPHA, diff=DIFF, omax=OMAX, k=K)
    motifs = miner.mine(ts)
    plot_motifs([axs[2]], ts, [motifs[2]], fn="tetris")


def get_data(column):
    if column == "ear":
        path = join(DIR, "ear")
        files = sorted(listdir(path))
        ts = []
        subjects = []
        for file in files:
            if file[:10] == EXCLUDE:
                continue
            row = np.loadtxt(join(path, file), delimiter=",", skiprows=1, usecols=1)
            if OBS:
                row = row[:OBS]
            ts.append(row)
            subjects.append(file[:10])
    else:
        file = join(DIR, f"{column}.csv")
        subjects = np.loadtxt(file, delimiter=",", usecols=0, skiprows=1, dtype="str")
        exclude = np.where(subjects == EXCLUDE)[0][0]
        ts = np.loadtxt(file, delimiter=",", usecols=range(1, OBS + 1), skiprows=1)
        ts = np.vstack((ts[:exclude], ts[exclude + 1 :]))
        subjects = np.hstack((subjects[:exclude], subjects[exclude + 1 :]))
    return np.array(ts), np.array(subjects)


if __name__ == "__main__":
    main()
