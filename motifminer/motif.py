from dataclasses import dataclass
from collections import defaultdict

import numpy as np

@dataclass
class Motif:
    pattern: str
    indexes: defaultdict(list)
    occurences: np.ndarray
    representative: np.ndarray

    def __init__(self, pattern):
        self.pattern = pattern
        self.indexes = defaultdict(list)
        self.occurences = np.array([])
        self.representative = np.array([])
    
    def record_index(self, i, j):
        """Record starting index of pattern in sequence i at position j."""
        self.indexes[i].append(j)