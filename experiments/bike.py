import json
from os import listdir
from os.path import join
from datetime import datetime

import matplotlib.pyplot as plt

from motifminer.miner import Miner
from motifminer.preprocessing import breakpoints
from quantitative import remove_spines

JSON_DIR = 'bike'
FIELD = 'heart_rate'
PARSE = False

# Miner parameters
MIN_SUP = 0.5
SEGMENT = 15
ALPHABET = 5
MIN_LEN = 3
MAX_OVERLAP = 0.7
LOCAL = True
K = 4


def main():
    records = get_records(JSON_DIR, PARSE)
    field = get_fields(records, FIELD)

    miner = Miner(field, MIN_SUP, SEGMENT, ALPHABET, MIN_LEN, MAX_OVERLAP, LOCAL, K)
    motifs = miner.mine_motifs()

    plot_motifs(motifs)

    # Find coordinate of motif occurrences
    # lat = get_fields(records, 'position_lat')
    # long = get_fields(records, 'position_long')
    # motif = motifs[0]
    #
    # lat_matches = []
    # long_matches = []
    # for ts, (start, end) in motif.match_indexes.items():
    #     # Just match beginning of motif
    #     lat_matches.append(lat[ts][start])
    #     long_matches.append(long[ts][start])
    #
    # coord_matches = list(zip(lat_matches, long_matches))
    # for coord in coord_matches:
    #     print(coord)
    #
    # print(len(coord_matches))
    # plt.scatter(long_matches, lat_matches)
    # plt.axis('square')
    # plt.show()

    
def get_records(directory, parse):
    """Get records from JSON files and optionally convert timestamp to datetime."""
    records = []
    for file in listdir(directory):
        with open(join(directory, file)) as fp:
            records.append(json.load(fp))

    if parse:
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


def plot_motifs(motifs):
    fig, axs = plt.subplots(ncols=4, figsize=(16, 4))
    fig.set_dpi(300)
    for motif, ax in zip(motifs, axs):
        for match in motif.matches:
            ax.plot(match, 'k', lw=0.1)
        ax.plot(motif.representative, 'b', lw=1)
        ax.hlines(breakpoints[ALPHABET].values(), 0, len(motif.representative), 'k', lw=0.3)
        ax.set(ylim=(-2, 2), xticks=[0, len(motif.representative)], yticks=[])
        remove_spines(ax)
    plt.show()


if __name__ == '__main__':
    main()
