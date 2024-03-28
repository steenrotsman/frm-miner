from collections import defaultdict
from warnings import catch_warnings, simplefilter

import numpy as np


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
        """Map representative, matches, and naed using occurrences."""
        self._seglen = seglen
        self.length = len(self.pattern) * seglen
        self.set_average_occurrences(ts)
        self.set_representative()
        self.set_best_matches_and_distance(ts, p)

    def set_average_occurrences(self, ts):
        for i, indexes in self.get_all_indexes().items():
            occurrences = []
            for index in indexes:
                occurrence = self.get_occurrence(ts[i], index)
                occurrences.append(occurrence)
            with catch_warnings():
                simplefilter("ignore")
                self.average_occurrences[i] = np.nanmean(occurrences, axis=0)

    def get_occurrence(self, ts, index):
        start = index * self._seglen
        end = start + self.length

        # Ensure motif occurrences are all the same length
        too_short = max(0, end - len(ts))
        return np.hstack((ts[start:end], np.array(too_short * [np.nan])))

    def set_representative(self):
        self.representative = np.nanmean(
            [ao for ao in self.average_occurrences.values()], axis=0
        )

    def set_best_matches_and_distance(self, ts, p):
        for i, indexes in self.get_all_indexes().items():
            best_match = 0
            min_dist = np.inf

            for index in indexes:
                occurrence = self.get_occurrence(ts[i], index)
                dist = np.linalg.norm(occurrence - self.representative)

                if dist < min_dist:
                    min_dist = dist
                    best_match = index
            self.distance += min_dist
            self.best_matches[i] = best_match * self._seglen
        self.distance /= len(self.indexes) * self.length ** (1 / p)
