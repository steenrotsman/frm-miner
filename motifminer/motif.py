from collections import defaultdict

import tensorflow as tf
from tensorflow.math import reduce_mean as m

class Motif:
    def __init__(self, pattern):
        self.pattern = pattern
        self.indexes = defaultdict(list)
        self.occurences = {}
        self.representative = None
        self.matches = []
        self.match_indexes = {}
        self.rmse = 0.0
    
    def record_index(self, i, j):
        """Record starting index of pattern in sequence i at position j."""
        self.indexes[i].append(j)
    
    def map(self):
        """Map representative, matches, and rmse using occurences."""
        self.representative = m([m(o, 0) for o in self.occurences.values()], 0)

        for seq, occurences in self.occurences.items():
            best_match = None
            best_match_index = []
            min_rmse = 100
            for i, occurence in enumerate(occurences):
                rmse = m((occurence - self.representative) ** 2) ** 0.5
                if rmse < min_rmse:
                    best_match = occurence
                    min_rmse = rmse
                    idx = self.indexes[seq][i]
                    best_match_index = [idx, idx + len(self.pattern)]
            self.rmse += min_rmse
            self.matches.append(best_match)
            self.match_indexes[seq] = best_match_index
        self.rmse /= len(self.match_indexes)
        self.matches = tf.stack(self.matches)
