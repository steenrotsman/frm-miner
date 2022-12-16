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
        Dictionary with patterns as keys and Motifs as values.
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
        self.mine_1_patterns()
        
        # There are no 0-patterns and we just mined 1-patterns, set k=2
        k = 2

        # If there were no frequent k-patterns, there can be no 
        # frequent (k+1)-patterns; stop
        while self._L[k-1] != []:
            self._L.append([])
            C = self.get_candidates(self._L[k-1])

            for a in C:
                # Count candidate occurences
                self.frequent[a] = Motif(a)

                # Find candidate in sequences parent occurs in
                for seq, indexes in self.frequent[a[:-1]].indexes.items():
                    sequence = self.sequences[seq]
                    for index in indexes:
                        if sequence[index : index+k] == a:
                            self.frequent[a].record_index(seq, index)
                
                # Check if candidate complies with minimum support
                self.prune_infrequent(a)

            k += 1        
    
    def prune_trivial(self, l: int = 3, o: float = 0.9):
        """Prune trivial patterns.
        
        Not all frequent patterns that comply with the minimum support
        are interesting. For example, 1-patterns are too trivial to be
        considered interesting.
        Furthermore, mining only maximal patterns can leave patterns
        that are trivial because they are almost contained in other
        patterns. For instance, if aaaaaa is a frequent pattern, aaaaba
        would still be a maximal pattern, as it differs by an item.
        However, it could still be considered trivial, as almost all of
        its contents are contained in the longer pattern.

        Parameters
        ----------
        l : int, optional.
            Minimum length of a pattern to be considered interesting.
        o : float, optional.
            Maximal redundancy ratio, calculated by
            len(longest common substring) / len(shortest pattern).
            If the redundancy ratio exceeds `o`, the shortest pattern is
            pruned.
        """
        # Prune from frequent patterns
        self.frequent = {k: v for k, v in self.frequent.items() if len(k) >= l}
        
        # Prune from maximal patterns
        self.maximal = {k: v for k, v in self.maximal.items() if len(k) >= l}

        # Prune patterns for which x% is overlapping from maximal
        patterns = list(self.maximal)
        patterns.sort(key=lambda x: (len(x), x), reverse=True)
        pruned = []
        
        for p1 in patterns:
            for p2 in patterns:
                # Only check unseen patterns and pairs
                if len(p2) >= len(p1) or p1 in pruned or p2 in pruned:
                    continue

                # Find longest common substring
                n, m = len(p1), len(p2)
                l = self.lcs(p1, p2, n, m)

                # Check if shorter pattern consists mostly of lcs
                if l / m > o:
                    self.maximal.pop(p2, 0)
                    pruned.append(p2)

    def lcs(self, p1: str, p2: str, n: int, m: int)-> int:
        """Longest common substring.
        
        Find the length of the longest string that is contained in two
        patterns. Uses dynamic programming algorithm adapted from
        https://www.geeksforgeeks.org/longest-common-substring-dp-29/

        Parameters
        ----------
        p1 : str
            String representation of first pattern.
        p2 : str
            String representation of second pattern.
        n : int
            Length of first pattern.
        m : int
            Length of second pattern.
        
        Returns
        -------
        res : int
            Length of the longest common substring of `p1` and `p2`.
        """
        dp = [[0 for row in range(m + 1)] for col in range(2)]
        res = 0
        
        for row in range(1, n + 1):
            for col in range(1, m + 1):
                if(p1[row - 1] == p2[col - 1]):
                    dp[row % 2][col] = dp[(row - 1) % 2][col - 1] + 1
                    if(dp[row % 2][col] > res):
                        res = dp[row % 2][col]
                else:
                    dp[row % 2][col] = 0
        return res
    
    def get_candidates(self, patterns: list[str])-> set[str]:
        """Use frequent (k-1)-patterns to generate candidate k-patterns.
        
        Candidates are generated by joining two patterns from L[k-1]
        if pattern1[1:] == pattern2[:-1].

        Parameters
        ----------
        patterns : list[str]
            List of frequent patterns that have a length of k - 1.

        Returns
        -------
        C : set[str]
            Set of unique candidate patterns that have a length of k.
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
    
    def mine_1_patterns(self):
        """Mine frequent 1-patterns.
        
        One scan is made over the sequences to map 1-pattern indexes.
        """
        # Divide sequences into indexes
        for i, sequence in enumerate(self.sequences):
            for j, item in enumerate(sequence):
                if item not in self.frequent:
                    self.frequent[item] = Motif(item)
                self.frequent[item].record_index(i, j)

        # Prune infrequent patterns
        for a in list(self.frequent.keys()):
            self.prune_infrequent(a)
    
    def prune_infrequent(self, a: str):
        """Prune infrequent patterns.
        
        - Prunes patterns with a too low support;
        - Adds frequent patterns to list of frequent patterns
        - Adds frequent patterns to maximal pattern indexes
        - Removes frequent pattern parents from maximal pattern indexes

        Parameters
        ----------
        a : str
            The pattern that should be processed.
        """
        if len(self.frequent[a].indexes) < self._min_freq:
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
