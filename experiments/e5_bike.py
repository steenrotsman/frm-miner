import json
from itertools import chain
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
from plot import plot_motifs
from scipy.stats import zscore

from frm import Miner

JSON_DIR = 'bike'
FIELD = 'speed'

# Miner parameters
MINSUP = 0.3
SEGLEN = 15
ALPHABET = 5
OMAX = [0.9, 0.8]
P = 1
K = 6


def main():
    records = get_records(JSON_DIR)
    field = get_fields(records, FIELD)
    data = [zscore(ts) for ts in field]

    for omax in OMAX:
        fig, axs = plt.subplots(ncols=3, nrows=2, sharey='all')
        miner = Miner(MINSUP, SEGLEN, ALPHABET, max_overlap=omax, k=K, p=P)
        motifs = miner.mine(field)
        plot_motifs(chain.from_iterable(axs), data, motifs, fn=f'5 bike motifs {omax}')


def get_records(directory):
    records = []
    for file in listdir(directory):
        with open(join(directory, file)) as fp:
            records.append(json.load(fp))

    return records


def get_fields(records, field_name):
    fields = []
    for record in records:
        if field_name in record[0]:
            field = [rec[field_name] for rec in record if rec[field_name] is not None]
            fields.append(field)

    return fields


if __name__ == '__main__':
    main()
