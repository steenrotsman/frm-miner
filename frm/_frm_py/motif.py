from collections import defaultdict
from warnings import catch_warnings, simplefilter

import numpy as np


class Motif:
    def __init__(self, pattern):
        self.pattern = pattern
        self.indexes = defaultdict(list)
        self.occurrences = {}
        self.average_occurrences = {}
        self.representative = None
        self.matches = []
        self.best_matches = {}
        self.naed = 0.0
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
    
    def map(self, ts, seglen):
        """Map representative, matches, and naed using occurrences."""
        self._seglen = seglen
        self.length = len(self.pattern) * seglen
        self.set_average_occurrences(ts)
        self.set_representative()
        self.set_best_matches_and_naed(ts)

    def set_average_occurrences(self, ts):
        for i, indexes in self.indexes.items():
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
        if too_short := max(0, end - len(ts)):
            return np.hstack((ts[start : end], np.array(too_short * [np.nan])))
        return ts[start : end]

    def set_representative(self):
        self.representative = np.nanmean([ao for ao in self.average_occurrences.values()], axis=0)

    def set_best_matches_and_naed(self, ts):
        for i, indexes in self.indexes.items():
            best_match = 0
            min_naed = 10 ** 6

            for index in indexes:
                occurrence = self.get_occurrence(ts[i], index)
                naed = np.nansum((occurrence - self.representative) ** 2) ** 0.5
                if naed < min_naed:
                    min_naed = naed
                    best_match = index
            self.naed += min_naed
            self.best_matches[i] = best_match * self._seglen
        self.naed /= (len(self.indexes)) * (self.length)
