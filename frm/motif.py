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
        if type(other) is not type(self):
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

    def map(self, ts, seglen, max_dist):
        """Map representative, matches, and distance using occurrences."""
        self._seglen = seglen
        self._ts = ts
        self.length = len(self.pattern) * seglen

        self.set_best_matches_and_distance(max_dist)
        self.trim_length(max_dist)
        self.set_representative()

    def set_best_matches_and_distance(self, max_dist):
        """Select occurrence with minimal radii and calculate extent."""
        for ts_index, start_indexes in self.get_all_indexes().items():
            min_radius = np.inf
            for start_index in start_indexes:
                radius = self.get_radius(ts_index, start_index)
                if radius < min_radius:
                    min_radius = radius
                    self.best_matches[ts_index] = start_index
            self.distance = max(self.distance, min_radius)

    def trim_length(self, max_dist):
        """Trim length of occurrences if beneficial."""
        occurrences = np.array(
            [
                self.get_occurrence(ts_index, start_index)
                for ts_index, start_index in self.best_matches.items()
            ]
        )

        # Find stable begin and end points of occurrences
        diff = np.max(occurrences, axis=0) - np.min(occurrences, axis=0)
        reference = np.mean(diff[self._seglen : -self._seglen])
        left_trim = np.where(diff[: self._seglen] <= reference)[0]
        left_trim = left_trim[0] if left_trim.size > 0 else 0
        right_trim = np.where(diff[-self._seglen :] <= reference)[0]
        right_trim = self._seglen - right_trim[-1] if right_trim.size > 0 else 0

        # Apply left and right trim
        for ts_index, start_index in self.best_matches.items():
            self.best_matches[ts_index] = (start_index * self._seglen) + left_trim
        self.length = len(self.pattern) * self._seglen - left_trim - right_trim

        # Recalculate extent
        self.distance = 0
        for (a, i), (b, j) in permutations(self.best_matches.items(), 2):
            occ1 = self.pad(znorm(self._ts[a][i : i + self.length]))
            occ2 = self.pad(znorm(self._ts[b][j : j + self.length]))
            self.distance = max(self.distance, ED(occ1, occ2))
        self.distance /= self.length ** (1 / 2)

    def set_representative(self):
        """Set representative motif as stepwise average of occurrences."""
        with catch_warnings():
            simplefilter("ignore")
            self.representative = np.nanmean(
                [
                    self.pad(
                        self._ts[ts_index][start_index : start_index + self.length]
                    )
                    for ts_index, start_index in self.best_matches.items()
                ],
                axis=0,
            )
        self.representative = self.representative[~np.isnan(self.representative)]
        self.length = len(self.representative)

    def get_occurrence(self, ts_index, start_index):
        """Get occurrence from time series with padding if needed."""
        start = start_index * self._seglen
        end = start + self.length

        return self.pad(self._ts[ts_index][start:end])

    def pad(self, ts):
        """Ensure occurrences are all the same length."""
        short = self.length - len(ts)
        return np.hstack((ts, np.array(short * [np.nan])))

    def get_radius(self, ts_index, start_index):
        """Maximum distance between pattern and nearest match in each ts."""
        radius = 0
        occ = znorm(self.get_occurrence(ts_index, start_index))
        for i, indexes in self.get_all_indexes().items():
            if i == ts_index:
                continue
            dist = min(ED(occ, znorm(self.get_occurrence(i, idx))) for idx in indexes)
            radius = max(radius, dist)

        return radius

    def get_more_matches(self):
        """Find matches in time series without matches if radius is not too high."""
        a = {}
        for i, series in enumerate(self._ts):
            if i not in self.best_matches:
                with catch_warnings():
                    simplefilter("ignore")
                    m = mass(self._ts[i], self.representative)

                best = np.argmin(m)
                radius = 0
                occ = znorm(self._ts[i][best : best + self.length])
                for j, indexes in self.get_all_indexes().items():
                    if i == j:
                        continue
                    dist = min(
                        ED(occ, znorm(self.get_occurrence(j, idx))) for idx in indexes
                    )
                    radius = max(radius, dist)
                radius /= self.length ** (1 / 2)
                if radius < self.distance:
                    self.best_matches[i] = best
                    a[i] = best


def ED(a, b):
    """Euclidean distance. Note: a and b need to be normalised beforehand."""
    return np.sqrt(np.nansum(np.square(a - b)))


znorm = partial(zscore, nan_policy="omit")
