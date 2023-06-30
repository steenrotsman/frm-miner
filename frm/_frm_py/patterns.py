"""Patterns module.

This module defines the PatternMiner class, a class that takes a collection of
sequences and mines frequent and maximal patterns from it. The patterns
can be filtered to have a certain minimum length.
"""
from .motif import Motif


class PatternMiner:
    """Mine patterns from a collection of sequences with PatternMiner.

    Parameters
    ----------
    minsup : float
        The minimum support for a pattern.
    
    Attributes
    ----------
    frequent : dict
        Dictionary of frequent motifs with string patterns as keys and Motif objects as values.
    """
    def __init__(
            self,
            minsup: float = 0.5,
            min_len: int = 3,
            max_len: int = 0,
            max_overlap: float = 0.9):
        self.minsup = minsup
        self.min_len = min_len
        self.max_len = max_len
        self.max_overlap = max_overlap

        self.frequent = {}

        # Frequency is easier to check than support
        self._min_freq = 0

        # Keep track of length we're currently mining
        self._k = 2

        # self.patterns[k] contains a list of patterns of length k
        self._patterns = [[], []]

    def mine(self, sequences):
        """Mine sequence motifs.

        Starting with patterns of length 1 (1-patterns), candidates are
        generated and checked for sufficient support. Then, k-patterns
        are generated based on (k-1)-patterns.

        sequences : list
            Collection of sequences with discrete values.
        """
        self._min_freq = len(sequences) * self.minsup

        # Mine 1-patterns separately from longer patterns
        self.mine_1_patterns(sequences)

        # If there were no frequent k-patterns, there can be no 
        # frequent (k+1)-patterns; stop
        while self._patterns[self._k - 1] and (not self.max_len or self._k <= self.max_len):
            self._patterns.append([])

            for candidate in self.get_candidates():
                # Count candidate occurrences
                self.frequent[candidate] = Motif(candidate)

                # Find candidate in sequences parent occurs in
                for seq, indexes in self.frequent[candidate[:-1]].indexes.items():
                    sequence = sequences[seq]
                    for index in indexes:
                        if sequence[index : index+self._k] == candidate:
                            self.frequent[candidate].record_index(seq, index)
                
                # Check if candidate complies with minimum support
                self.prune_infrequent(candidate)

            self._k += 1

        self.remove_redundant()
    
    def remove_redundant(self):
        """Remove redundant patterns.
        
        Not all frequent patterns that comply with the minimum support
        are interesting. For example, 1-patterns are too trivial to be
        considered interesting.
        Furthermore, mining only maximal patterns can leave patterns
        that are trivial because they are almost contained in other
        patterns. For instance, if aaaaaa is a frequent pattern, aabaa
        could be considered trivial, as almost all of its contents are
        contained in the longer pattern.

        Parameters
        ----------
        min_len : int, optional.
            Minimum length of a pattern to be considered interesting.
        o : float, optional.
            Maximal redundancy ratio, calculated by
            length(longest common subsequence) / length(shortest pattern).
            If the redundancy ratio exceeds `max_overlap`, the shortest pattern is
            pruned.
        """
        # Prune from frequent patterns
        self.frequent = {k: v for k, v in self.frequent.items() if len(k) >= self.min_len}

        # If max_overlap == 1, no overlap is too high
        if self.max_overlap == 1:
            return

        # Prune patterns for which x% is overlapping from frequent
        patterns = list(self.frequent)
        patterns.sort(key=lambda x: (len(x), x), reverse=True)
        pruned = []
        
        for p1 in patterns:
            if p1 in pruned:
                continue
            for p2 in patterns:
                n, m = len(p1), len(p2)
                # Only check unseen patterns and pairs
                if m > n or p1 == p2 or p2 in pruned:
                    continue

                # Check if shorter pattern consists mostly of lcs
                if self.lcs(p1, p2, n, m) / m > self.max_overlap:
                    self.frequent.pop(p2, 0)
                    pruned.append(p2)

    def lcs(self, p1: str, p2: str, n: int, m: int) -> int:
        """Longest common subsequence.
        
        Find the length of the longest sequence that is contained in two
        patterns. Uses dynamic programming algorithm adapted from
        https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/

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
            Length of the longest common subsequence of `p1` and `p2`.
        """
        # declaring the array for storing the dp values
        L = [[0]*(m+1) for i in range(n+1)]

        """Following steps build L[n+1][m+1] in bottom up fashion
        Note: L[i][j] contains length of LCS of X[0..i-1]
        and Y[0..j-1]"""
        for i in range(n+1):
            for j in range(m+1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif p1[i-1] == p2[j-1]:
                    L[i][j] = L[i-1][j-1]+1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])

        # L[n][m]] contains the length of LCS of X[0..n-1] & Y[0..m-1]
        return L[n][m]
    
    def get_candidates(self) -> list[str]:
        """Use frequent (k-1)-patterns to generate candidate k-patterns.
        
        Candidates are generated by joining two patterns from L[k-1]
        if pattern_1[1:] == pattern_2[:-1].

        Returns
        -------
        candidates : list[str]
            Set of candidate patterns that have a length of k.
        """
        patterns = self._patterns[self._k - 1]
        return [p1 + p2[-1] for p1 in patterns for p2 in patterns if p1[1:] == p2[:-1]]
    
    def mine_1_patterns(self, sequences):
        """Mine frequent 1-patterns.
        
        One scan is made over the sequences to map 1-pattern indexes.

        sequences : list
            Collection of sequences with discrete values.
        """
        # Divide sequences into indexes
        for i, sequence in enumerate(sequences):
            for j, item in enumerate(sequence):
                if item not in self.frequent:
                    self.frequent[item] = Motif(item)
                self.frequent[item].record_index(i, j)

        # Prune infrequent patterns
        for a in list(self.frequent.keys()):
            self.prune_infrequent(a)
    
    def prune_infrequent(self, pattern: str):
        """Prune infrequent patterns.
        
        - Prunes patterns with a too low support;
        - Adds frequent patterns to list of frequent patterns

        Parameters
        ----------
        pattern : str
            The pattern that should be processed.
        """
        if len(self.frequent[pattern].indexes) < self._min_freq:
            # Delete indexes of this infrequent pattern
            self.frequent.pop(pattern)
        else:
            # Add to list of frequent k-patterns
            self._patterns[len(pattern)].append(pattern)
