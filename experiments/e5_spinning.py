import matplotlib.pyplot as plt
from e5_bike import JSON_DIR, get_fields, get_records
from plot import plot_motifs
from scipy.stats import zscore

from frm import Miner

FIELD = 'heart_rate'

# Miner parameters
MINSUP = 0.3
SEGLEN = 15
ALPHABET = 5
P = 1
K = 3


def main():
    records = get_records(JSON_DIR)

    data = {
        'spinning': get_fields([r for r in records if not r[-1]['distance']], FIELD),
        'cycling': get_fields([r for r in records if r[-1]['distance']], FIELD),
    }

    for category, field in data.items():
        fig, axs = plt.subplots(ncols=K, sharey='all')
        miner = Miner(MINSUP, SEGLEN, ALPHABET, max_overlap=0.9, k=K, p=P)
        motifs = miner.mine(field)
        plot_motifs(
            axs,
            [zscore(ts) for ts in field],
            motifs,
            fn=f'5 {category} motifs',
            yticks=[-1, 0, 1],
        )


if __name__ == '__main__':
    main()
