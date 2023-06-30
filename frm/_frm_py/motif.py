from collections import defaultdict

import numpy as np


class Motif:
    def __init__(self, pattern):
        self.pattern = pattern
        self.indexes = defaultdict(list)
        self.occurrences = {}
        self.representative = None
        self.matches = []
        self.match_indexes = {}
        self.naed = 0.0

    def __repr__(self):
        return f"Motif('{self.pattern}')"

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return other.pattern == self.pattern

    def record_index(self, i, j):
        """Record starting index of pattern in sequence i at position j."""
        self.indexes[i].append(j)
    
    def map(self):
        """Map representative, matches, and naed using occurrences."""
        self.representative = np.mean([np.mean(o, 0) for o in self.occurrences.values()], 0)

        for seq, occurrences in self.occurrences.items():
            best_match = None
            best_match_index = []
            min_naed = 100
            for i, occurrence in enumerate(occurrences):
                naed = np.sum((occurrence - self.representative) ** 2) ** 0.5
                if naed < min_naed:
                    best_match = occurrence
                    min_naed = naed
                    idx = self.indexes[seq][i]
                    best_match_index = [idx, idx + len(self.pattern)]
            self.naed += min_naed
            self.matches.append(best_match)
            self.match_indexes[seq] = best_match_index
        self.naed /= len(self.match_indexes) * len(self.representative)
