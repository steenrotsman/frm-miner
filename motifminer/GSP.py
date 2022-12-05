"""GSP module.

This module defines the GSP class, a class that takes a collection of
sequences and mines frequent and maximal patterns from it. The patterns
can be filtered to have a certain minimum length.
"""
from collections import defaultdict
from typing import Iterable

from .motif import Motif

class GSP:
    """Mine patterns from a collection of sequences with GSP.

    Parameters
    ----------
    sequences : Iterable
        Collection of sequences with distinct values.
    alphabet_size : int
        Number of discrete elements present in the time series.
    min_sup : float
        The minimum support for a pattern.
    
    Attributes
    ----------
    indexes : dict
        Dictionary with patterns as keys and a list of Motifs as values.
    maximal : dict
        Same as indexes but drops patterns contained in another pattern.
    """
    def __init__(
            self,
            sequences: Iterable,
            min_sup: float = 0.5):
        self._min_freq = len(sequences) * min_sup
        self._L = [[], []]

        self.sequences = sequences
        self.frequent = {}
        self.maximal = {}

    def mine(self):
        """Mine Generalised Sequential Patterns.

        Starting with patterns of length 1 (1-patterns), candidates are
        generated and checked for sufficient support. Then, k-patterns
        are generated based on (k-1)-patterns.
        """
        # Mine 1-patterns separately from longer patterns
        self._mine_1_patterns()
        
        # There are no 0-patterns and we just mined 1-patterns, set k=2
        k = 2

        # If there were no frequent k-patterns, there can be no 
        # frequent (k+1)-patterns; stop
        while self._L[k-1] != []:
            self._L.append([])
            C = self._get_candidates(self._L[k-1])

            for a in C:
                # Count candidate occurences
                self.frequent[a] = defaultdict(list)

                # Find candidate in sequences parent occurs in
                for seq, indexes in self.frequent[a[:-1]].items():
                    sequence = self.sequences[seq]
                    for index in indexes:
                        if sequence[index : index+k] == a:
                            self.frequent[a][seq].append(index)
                
                # Check if candidate complies with minimum support
                self._prune(a)

            k += 1        
    
    def prune(self, l: int = 3):
        """Prune too short patterns.
        
        Not all frequent patterns that comply with the minimum support
        are interesting. For example, 1-patterns are too trivial to be
        considered interesting.

        Parameters
        ----------
        l : int, optional.
            Minimum length of a pattern to be considered interesting.
        """
        # Prune from frequent patterns
        self.frequent = {k: v for k, v in self.frequent.items() if len(k) >= l}
        
        # Prune from maximal patterns
        self.maximal = {k: v for k, v in self.maximal.items() if len(k) >= l}
    
    def _get_candidates(self, patterns: list[str])-> set[str]:
        """Use frequent (k-1)-patterns to generate candidate k-patterns.
        
        Candidates are generated by joining two patterns from L[k-1]
        if pattern1[1:] == pattern2[:-1].
        """
        # Ensure only unique patterns are generated
        C = set()

        for pattern1 in patterns:
            for pattern2 in patterns:
                # Check if patterns can be joined
                if pattern1[1:] == pattern2[:-1]:
                    # Add candidate pattern that joins the two patterns
                    C.add(pattern1 + pattern2[-1])
        
        return C
    
    def _mine_1_patterns(self):
        """Mine frequent 1-patterns."""
        # Divide sequences into indexes
        for i, sequence in enumerate(self.sequences):
            for j, item in enumerate(sequence):
                self.frequent[item].record_index(i, j)

        # Prune infrequent patterns
        for a in list(self.frequent.keys()):
            self._prune(a)
    
    def _prune(self, a):
        """Prune infrequent patterns.
        
        - Prunes patterns with a too low support;
        - Adds frequent patterns to list of frequent patterns
        - Adds frequent patterns to maximal pattern indexes
        - Removes frequent pattern parents from maximal pattern indexes
        """
        if len(self.frequent[a]) < self._min_freq:
            # Delete indexes of this infrequent pattern
            self.frequent.pop(a)
        else:
            # Add to list of frequent k-patterns
            self._L[len(a)].append(a)

            # Add to maximal patterns
            self.maximal[a] = self.frequent[a]

            # The patterns joined to create a cannot be maximal patterns
            self.maximal.pop(a[:-1], 0)
            self.maximal.pop(a[1:], 0)
