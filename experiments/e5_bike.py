import json
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
from scipy.stats import zscore

from frm._frm_py.miner import Miner as PyMiner
from frm import Miner as CppMiner
from plot import plot_motifs

JSON_DIR = 'bike'
FIELD = 'speed'

# Miner parameters
MINSUP = 0.3
SEGLEN = 10
ALPHABET = 5
MAX_OVERLAP = [0.9, 0.7]
K = 0


def main():
    records = get_records(JSON_DIR)
    field = get_fields(records, FIELD)

    fig, axss = plt.subplots(nrows=len(MAX_OVERLAP), ncols=1, layout='compressed')
    for max_overlap, axs in zip(MAX_OVERLAP, axss):
        py_miner = PyMiner(MINSUP, SEGLEN, ALPHABET, max_overlap=max_overlap, k=K)
        cp_miner = CppMiner(MINSUP, SEGLEN, ALPHABET, max_overlap=max_overlap, k=K)

        py_motifs = py_miner.mine(field)
        cp_motifs = cp_miner.mine(field)
        plot_motifs(fig, axs, [zscore(ts) for ts in field], py_motifs, ALPHABET)
    plt.savefig(join('figs', '5 bike motifs.eps'))


def get_records(directory):
    """Get records from JSON files and optionally convert timestamp to datetime."""
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
