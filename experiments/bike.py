import json
from os import listdir
from os.path import join
from datetime import datetime

import matplotlib.pyplot as plt

from frm.miner import Miner
from plot import plot_motifs

JSON_DIR = 'bike'
FIELD = 'speed'
PARSE_TIMESTAMP = False

# Miner parameters
MIN_SUP = 0.5
SEGMENT = 15
ALPHABET = 5
MIN_LEN = 3
MAX_OVERLAP = 0.7
LOCAL = True
K = 4


def main():
    records = get_records(JSON_DIR, PARSE_TIMESTAMP)
    field = get_fields(records, FIELD)

    miner = Miner(field, MIN_SUP, SEGMENT, ALPHABET, MIN_LEN, MAX_OVERLAP, LOCAL, K)
    motifs = miner.mine_motifs()

    fig, axs = plt.subplots(ncols=4, layout='compressed')
    plot_motifs(fig, axs, motifs, ALPHABET)


def get_records(directory, parse_timestamp):
    """Get records from JSON files and optionally convert timestamp to datetime."""
    records = []
    for file in listdir(directory):
        with open(join(directory, file)) as fp:
            records.append(json.load(fp))

    if parse_timestamp:
        # Convert timestamp string to timestamp type
        for i, record in enumerate(records):
            for j, rec in enumerate(record):
                records[i][j]['timestamp'] = datetime.strptime(rec['timestamp'], '%Y-%m-%d %H:%M:%S%z')

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
