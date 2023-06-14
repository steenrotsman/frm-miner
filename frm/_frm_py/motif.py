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
        self.rmse = 0.0

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
        """Map representative, matches, and rmse using occurrences."""
        self.representative = np.mean([np.mean(o, 0) for o in self.occurrences.values()], 0)

        for seq, occurrences in self.occurrences.items():
            best_match = None
            best_match_index = []
            min_rmse = 100
            for i, occurrence in enumerate(occurrences):
                rmse = np.mean((occurrence - self.representative) ** 2) ** 0.5
                if rmse < min_rmse:
                    best_match = occurrence
                    min_rmse = rmse
                    idx = self.indexes[seq][i]
                    best_match_index = [idx, idx + len(self.pattern)]
            self.rmse += min_rmse
            self.matches.append(best_match)
            self.match_indexes[seq] = best_match_index
        self.rmse /= len(self.match_indexes)