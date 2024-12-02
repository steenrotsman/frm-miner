from collections import defaultdict
from functools import partial
from itertools import permutations
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.stats import zscore

with catch_warnings():
    simplefilter("ignore")
    from mass_ts import mass2 as mass


class Motif:
    def __init__(self, pattern):
        self.pattern = pattern
        self.indexes = defaultdict(list)
        self.children = []
        self.average_occurrences = {}
        self.representative = None
        self.best_matches = {}
        self.distance = 0.0
        self.length = 0
        self._seglen = 0
        self._ts = []

    def __repr__(self):
        return f"Motif('{self.pattern}')"

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return other.pattern == self.pattern

    def record_index(self, i, j):
        """Record starting index of pattern in sequence i at position j."""
        self.indexes[i].append(j)

    def remove_index(self, i, j):
        """Record starting index of pattern in sequence i at position j.

        Removes position j from starting indexes of pattern in sequence i.
        If j was the only starting index of pattern in sequence i, removes sequence i from indexes.
        """
        self.indexes[i].remove(j)
        if not len(self.indexes[i]):
            self.indexes.pop(i, 0)

    def get_all_indexes(self):
        """Get dict of all indexes of motif, including its children."""
        indexes = defaultdict(list)

        for seq, idx in self.indexes.items():
            indexes[seq] += idx

        for child in self.children:
            for seq, idx in child.get_all_indexes().items():
                indexes[seq] += idx

        return indexes

    def map(self, ts, seglen, p):
        """Map representative, matches, and distance using occurrences."""
        self._seglen = seglen
        self.length = len(self.pattern) * seglen
        self._ts = ts

        self.set_best_matches_and_distance()
        self.set_distance(p)
        self.set_representative()
        for i, index in self.best_matches.items():
            self.best_matches[i] *= seglen
        self.get_more_matches()

    def set_best_matches_and_distance(self):
        extent = 0
        for i, indexes in self.get_all_indexes().items():
            min_radius = np.inf
            for index in indexes:
                radius = self.get_radius(index, i)
                if radius < min_radius:
                    min_radius = radius
                    self.best_matches[i] = index
            if min_radius > extent:
                extent = min_radius

            # If there is just one occurrence in a time series, use that
            if len(indexes) == 1:
                self.best_matches[i] = indexes[0]
                continue
            # Select index with minimal radius
            radius = partial(self.get_radius, ts_index=i)
            self.best_matches[i] = min(indexes, key=radius)

    def set_distance(self, p):
        # TODO the extent is the maximum radius of any time series (or rather the two time series with the farthest apart occurrences)
        # Calculating extent could just be done in the set_best_matches method
        for (a, i), (b, j) in permutations(self.best_matches.items(), 2):
            occ1 = znorm(self.get_occurrence(a, i))
            occ2 = znorm(self.get_occurrence(b, j))
            distance = ED(occ1, occ2)
            self.distance = max(self.distance, distance)
        self.distance /= self.length ** (1 / p)

    def set_representative(self):
        with catch_warnings():
            simplefilter("ignore")
            self.representative = np.nanmean(
                [self.get_occurrence(p, i) for p, i in self.best_matches.items()],
                axis=0,
            )
        self.representative = self.representative[~np.isnan(self.representative)]
        self.length = len(self.representative)

    def get_occurrence(self, ts, index):
        start = index * self._seglen
        end = start + self.length

        # Ensure motif occurrences are all the same length
        too_short = max(0, end - len(self._ts[ts]))
        return np.hstack((self._ts[ts][start:end], np.array(too_short * [np.nan])))

    def get_radius(self, start_index, ts_index):
        radius = 0
        occ = zscore(self.get_occurrence(ts_index, start_index))
        for i, indexes in self.get_all_indexes().items():
            if i == ts_index:
                continue
            dist = min(ED(occ, znorm(self.get_occurrence(i, idx))) for idx in indexes)
            radius = max(radius, dist)

        return radius

    def get_more_matches(self):
        a = {}
        for i, series in enumerate(self._ts):
            if i not in self.best_matches:
                with catch_warnings():
                    simplefilter("ignore")
                    m = mass(self._ts[i], self.representative)

                best = np.argmin(m)
                radius = 0
                occ = zscore(self._ts[i][best : best + self.length])
                for j, indexes in self.get_all_indexes().items():
                    if i == j:
                        continue
                    dist = min(
                        ED(occ, znorm(self.get_occurrence(j, idx))) for idx in indexes
                    )
                    radius = max(radius, dist)
                if radius < self.distance:
                    self.best_matches[i] = best
                    a[i] = best
        a


def ED(a, b):
    return np.sqrt(np.nansum(np.square(a - b)))


znorm = partial(zscore, nan_policy='omit')
