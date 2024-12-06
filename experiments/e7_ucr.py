import random
from collections import defaultdict
from itertools import product
from multiprocessing import Pool
from os import listdir
from os.path import join

import numpy as np
from scipy.stats import zscore

from frm import Miner

FILE = 'e7_ucr.csv'
FOLDER = 'UCRArchive_2018'
FILES = listdir(FOLDER)
PARTITIONS = ['TRAIN', 'TEST']
THRESHOLD = 0.5

# Parameters for FRM-Miner
MINSUP = [0.3]
SEGLEN = range(10, 0, -1)
ALPHA = [3, 4, 5, 6]


def main():
    # First look if there are unfinished settings
    unfinished, total = get_unfinished()
    finish_settings(unfinished)

    # Then, generate more settings
    x = 1000 - total
    with Pool(processes=16, maxtasksperchild=1) as p:
        p.map(sim, range(x))


def sim(_):
    name = random.choice(FILES)
    data, motif_ts, motif_start, motif_length, injection_indices = get_sim_data(name)

    # Run FRM-Miner
    for minsup, seglen, alpha in product(MINSUP, SEGLEN, ALPHA):
        miner = Miner(minsup, seglen, alpha, omax=1)
        motifs = miner.mine(data)

        min_overlap = THRESHOLD * motif_length
        found = check(motifs, motif_ts, seglen, motif_start, motif_length, min_overlap)

        with open(FILE, 'a') as fp:
            fp.write(
                f'{name},{motif_ts},{motif_start},{motif_length},{len(injection_indices)},{minsup},{seglen},{alpha},{found}\n'
            )


def sim_setting(setting):
    experiment, (minsup, seglen, alpha) = setting
    data, ts, start, length, support = reload_sim_data(*experiment)

    miner = Miner(minsup, seglen, alpha, omax=1)
    motifs = miner.mine(data)

    min_overlap = THRESHOLD * length
    found = check(motifs, ts, seglen, start, length, min_overlap)

    with open(FILE, 'a') as fp:
        fp.write(
            f'{experiment[0]},{ts},{start},{length},{support},{minsup},{seglen},{alpha},{found}\n'
        )


def get_data(name):
    data = []
    for part in PARTITIONS:
        with open(join(FOLDER, name, f'{name}_{part}.tsv')) as f:
            for row in f:
                # Split data on tabs and parse to floats
                data.append(
                    np.array(
                        [
                            float(x)
                            for x in row.strip('\n').split('\t')[1:]
                            if x != 'NaN'
                        ]
                    )
                )

    return data


def get_sim_data(name):
    data = get_data(name)
    motif_ts = random.randint(0, len(data) - 1)
    motif_ts_len = len(data[motif_ts])
    motif_length = random.randint(int(0.05 * motif_ts_len), int(0.1 * motif_ts_len))
    motif_start = random.randint(0, motif_ts_len - motif_length)
    motif = zscore(data[motif_ts][motif_start : motif_start + motif_length])

    injection_indices = random.sample(range(len(data)), random.randint(1, len(data)))

    # If the origin of the motif is selected, remove it
    if motif_ts in injection_indices:
        injection_indices.remove(motif_ts)

    # Inject the motif into the selected time series at random positions
    for inject_index in injection_indices:
        inject_max = max(0, len(data[inject_index]) - motif_length)
        inject_start = random.randint(0, inject_max)
        inject_motif = data[inject_index][inject_start : inject_start + motif_length]
        inject_motif = np.mean(inject_motif) + np.std(inject_motif) * motif
        data[inject_index][inject_start : inject_start + motif_length] = inject_motif
    return data, motif_ts, motif_start, motif_length, injection_indices


def reload_sim_data(name, motif_ts, motif_start, motif_length, support, _):
    motif_ts = int(motif_ts)
    motif_start = int(motif_start)
    motif_length = int(motif_length)
    support = int(support)

    data = get_data(name)
    motif = zscore(data[motif_ts][motif_start : motif_start + motif_length])

    injection_indices = random.sample(range(len(data)), support + 1)

    # If the origin of the motif is selected, remove it
    if motif_ts in injection_indices:
        injection_indices.remove(motif_ts)
    else:
        injection_indices.pop()

    # Inject the motif into the selected time series at random positions
    for inject_index in injection_indices:
        inject_max = max(0, len(data[inject_index]) - motif_length)
        inject_start = random.randint(0, inject_max)
        inject_motif = data[inject_index][inject_start : inject_start + motif_length]
        inject_motif = np.mean(inject_motif) + np.std(inject_motif) * motif
        data[inject_index][inject_start : inject_start + motif_length] = inject_motif
    return data, motif_ts, motif_start, motif_length, len(injection_indices)


def check(motifs, motif_ts, seglen, motif_start, motif_length, min_overlap):
    for m in motifs:
        if motif_ts in m.get_all_indexes():
            for idx in m.get_all_indexes()[motif_ts]:
                occurrence_start = idx * seglen
                overlap_start = max(motif_start, occurrence_start)
                overlap_end = min(
                    motif_start + motif_length, occurrence_start + motif_length
                )
                overlap_length = max(0, overlap_end - overlap_start)
                if overlap_length >= min_overlap:
                    return True
    return False


def get_unfinished():
    experiments = defaultdict(list)

    with open(FILE) as fp:
        for row in fp:
            *experiment, minsup, seglen, alpha, found = row.strip().split(',')
            experiments[tuple(experiment)] = (minsup, seglen, alpha)
    total = len(experiments)
    for experiment, finished in experiments.copy().items():
        if len(finished) == len(MINSUP) * len(SEGLEN) * len(ALPHA):
            experiments.pop(experiment)

    return experiments, total


def finish_settings(unfinished):
    to_finish = []
    for settings in unfinished:
        for parameters in product(MINSUP, SEGLEN, ALPHA):
            if parameters not in unfinished[settings]:
                to_finish.append((settings, parameters))

    with Pool(16, maxtasksperchild=1) as p:
        p.map(sim_setting, to_finish)


if __name__ == '__main__':
    main()
