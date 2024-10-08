import json
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
from scipy.stats import zscore

from frm import Miner
from plot import plot_motifs

JSON_DIR = 'bike'
FIELD = 'speed'

# Miner parameters
MINSUP = 0.3
SEGLEN = 15
ALPHABET = 5
MAX_OVERLAP = [0.9]
K = 4


def main():
    records = get_records(JSON_DIR)
    field = get_fields(records, FIELD)

    fig, axss = plt.subplots(nrows=len(MAX_OVERLAP), ncols=K, sharey='all', layout='compressed')
    for max_overlap, axs in zip(MAX_OVERLAP, [axss]):
        miner = Miner(MINSUP, SEGLEN, ALPHABET, max_overlap=max_overlap, k=K)

        motifs = miner.mine(field)
        plot_motifs(fig, axs, [zscore(ts) for ts in field], motifs, ALPHABET)
    plt.savefig(join('figs', '5 bike motifs.eps'))


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
