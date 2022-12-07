from collections import defaultdict

import numpy as np

class Motif:
    def __init__(self, pattern):
        self.pattern = pattern
        self.indexes = defaultdict(list)
        self.occurences = []
        self.representative = np.array([])
        self.matches = []
        self.rmse = 0.0
    
    def record_index(self, i, j):
        """Record starting index of pattern in sequence i at position j."""
        self.indexes[i].append(j)
    
    def map(self):
        """Map representative, matches, and rmse using occurences."""
        self.representative = self.m([self.m(occ) for occ in self.occurences])
        
        self.matches = []
        self.rmse = 0.0

        for occurences in self.occurences:
            best_match = None
            min_rmse = 100
            for occurence in occurences:
                rmse = np.mean((occurence - self.representative) ** 2) ** 0.5
                if rmse < min_rmse:
                    best_match = occurence
                    min_rmse = rmse
            self.rmse += min_rmse
            self.matches.append(best_match)
        self.rmse /= len(self.matches)

    def m(self, a: list):
        """Wrapper around np.array and np.mean(axis=0)."""
        return np.mean(np.array(a), axis=0)