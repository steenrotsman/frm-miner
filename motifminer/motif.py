from collections import defaultdict

import numpy as np

class Motif:
    def __init__(self, pattern):
        self.pattern = pattern
        self.indexes = defaultdict(list)
        self.occurences = {}
        self.representative = np.array([])
        self.matches = {}
        self.match_indexes = {}
        self.rmse = 0.0
    
    def record_index(self, i, j):
        """Record starting index of pattern in sequence i at position j."""
        self.indexes[i].append(j)
    
    def map(self):
        """Map representative, matches, and rmse using occurences."""
        self.representative = self.m([self.m(o) for o in self.occurences.values()])

        for seq, occurences in self.occurences.items():
            best_match = None
            best_match_index = []
            min_rmse = 100
            for i, occurence in enumerate(occurences):
                rmse = np.mean((occurence - self.representative) ** 2) ** 0.5
                if rmse < min_rmse:
                    best_match = occurence
                    min_rmse = rmse
                    idx = self.indexes[seq][i]
                    best_match_index = [idx, idx + len(self.pattern)]
            self.rmse += min_rmse
            self.matches[seq] = best_match
            self.match_indexes[seq] = best_match_index
        self.rmse /= len(self.matches)

    def m(self, a: list):
        """Wrapper around np.array and np.mean(axis=0)."""
        return np.mean(np.array(a), axis=0)