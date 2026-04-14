import json
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
from plot import plot_motifs
from scipy.stats import zscore

from frm import Miner

JSON_DIR = "bike"
FIELD = "speed"

# Miner parameters
MINSUP = 0.3
SEGLEN = 6
ALPHABET = 4
K = 4


def main():
    records = get_records(JSON_DIR)
    field = get_fields(records, FIELD)
    data = [zscore(ts) for ts in field]

    fig, axs = plt.subplots(ncols=4, sharey="all")
    miner = Miner(MINSUP, SEGLEN, ALPHABET, k=K)
    motifs = miner.mine(field)
    plot_motifs(axs, data, motifs, fn="Fig6", yticks=[-3, 0, 3])


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
            if any(field):
                fields.append(field)

    return fields


if __name__ == "__main__":
    main()
