from multiprocessing import Pool
from os.path import join

import numpy as np
from e3_accuracy import get_data

from frm import Miner

SIMS = 15
K = 10

FOLDER = "experiments"
FILE = "e3_quantitative.csv"


def main():
    seen = []
    with open(join(FOLDER, FILE)) as fp:
        for line in fp:
            seen.append(int(line.split(",")[0]))
    with Pool(8) as p:
        list(p.imap_unordered(sim, [i for i in range(SIMS) if i not in seen]))


def sim(seed):
    # Randomly determine settings
    rng = np.random.default_rng(seed)
    minsup = rng.uniform(0.02, 1.0)
    seglen = rng.integers(1, 100)
    alpha = rng.integers(3, 10)
    omax = rng.uniform(0.5, 1.0)
    inject = rng.integers(1, 100)
    noise = rng.uniform(0.0, 2.0)
    print(f"{seed},{minsup},{seglen},{alpha},{omax},{inject},{noise}")

    # Generate data with injected motifs
    data, ground_truth = get_data(noise, rng)

    # Discover motifs
    mm = Miner(minsup, seglen, alpha, omax, k=K, diff=1)
    motifs = mm.mine(data)
    result = check(motifs, ground_truth)

    with open(join(FOLDER, FILE), "a") as fp:
        fp.write(f"{seed},{minsup},{seglen},{alpha},{omax},{inject},{noise},{result}\n")


def check(motifs, ground_truth):
    # Handle input if motifs is a single Motif
    if not isinstance(motifs, list):
        motifs = [motifs]

    # Find first motif with at least inject / 2 occurrences with sufficient overlap
    for i, motif in enumerate(motifs):
        minsup = len([x for x in ground_truth if x != -1]) / 2
        minsup = len(motif.best_matches) / 2
        min_overlap = motif.length / 2
        true_positives = 0
        for ts_index, start_index in motif.best_matches.items():
            # Check against ground truth
            overlap_start = max(ground_truth[ts_index], start_index)
            overlap_end = min(
                ground_truth[ts_index] + motif.length, start_index + motif.length
            )
            overlap_length = max(0, overlap_end - overlap_start)
            if overlap_length >= min_overlap:
                true_positives += 1
                if true_positives >= minsup:
                    return i

    # None of the top motifs corresponds to the ground truth
    return -1


if __name__ == "__main__":
    main()
