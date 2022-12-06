from dataclasses import dataclass
from collections import defaultdict

import numpy as np

@dataclass
class Motif:
    pattern: str
    indexes: defaultdict(list)
    occurences: list
    representative: np.ndarray
    matches: list
    rmse: float

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