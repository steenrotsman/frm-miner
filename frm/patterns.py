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
    omax : float
        The maximum overlap with longer patterns to not be considered redundant.

    Attributes
    ----------
    frequent : dict
        Dictionary of frequent motifs with string patterns as keys and Motif objects as values.
    """

    def __init__(self, minsup, omax=0.8):
        self.minsup = minsup
        self.omax = omax

        self.frequent = {}

        # Frequency is easier to check than support
        self._min_freq = 0

        # Keep track of length we're currently mining
        self._k = 2

        # Keep track of the patterns of length k-1 and length k
        self._patterns = [set(), set()]

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

        # If there were no frequent k-patterns, there can be no frequent (k+1)-patterns
        while self._patterns[1]:
            self._patterns = [self._patterns[1], set()]
            self.generate_candidates_from_parents(sequences)
            self.prune_infrequent()
            self._k += 1
        self.remove_redundant()

    def mine_1_patterns(self, sequences):
        """Make one scan over sequences to find frequent 1-patterns."""
        for i, sequence in enumerate(sequences):
            for j, item in enumerate(sequence):
                if item not in self.frequent:
                    self.frequent[item] = Motif(item)
                    self._patterns[1].add(item)
                self.frequent[item].record_index(i, j)
        self.prune_infrequent()

    def prune_infrequent(self):
        """Prune infrequent patterns.

        - Prunes patterns with a too low support;
        - Adds frequent patterns to list of frequent patterns
        """
        for pattern in self._patterns[1].copy():
            # Check if pattern occurs in enough time series to comply with minsup
            if len(self.frequent[pattern].indexes) < self._min_freq:
                self.frequent.pop(pattern)
                self._patterns[1].discard(pattern)
            # Reorder the tree for k > 1 patterns
            elif len(pattern) > 1:
                parent = self.frequent[pattern[:-1]]

                # Add candidate to parent's children
                parent.children.append(self.frequent[pattern])

                # Remove candidate indexes from parents
                for seq, indexes in self.frequent[pattern].indexes.items():
                    for index in indexes:
                        parent.remove_index(seq, index)

    def generate_candidates_from_parents(self, sequences):
        """Use frequent k-1 patterns to find k-pattern candidates."""
        for parent in self._patterns[0]:
            for seq, indexes in self.frequent[parent].indexes.items():
                for index in indexes:
                    candidate = sequences[seq][index : index + self._k]

                    # Consider only long enough candidates with two frequent parents
                    if candidate[1:] not in self.frequent or len(candidate) < self._k:
                        continue

                    # Keep track of new candidates
                    if candidate not in self.frequent:
                        self.frequent[candidate] = Motif(candidate)
                        self._patterns[1].add(candidate)

                    self.frequent[candidate].record_index(seq, index)

    def remove_redundant(self):
        """Remove redundant patterns."""
        # If max_overlap >= 1, no overlap is too high
        if self.omax >= 1:
            return

        # Remove patterns with too much overlap
        patterns = sorted(self.frequent, key=len, reverse=True)
        pruned = set()

        for p1 in patterns:
            if p1 in pruned:
                continue
            for p2 in patterns:
                n, m = len(p1), len(p2)
                # Only check unseen patterns and pairs
                if m >= n or p2 in pruned:
                    continue

                # Check if shorter pattern consists mostly of lcs
                if self.lcs(p1, p2, n, m) / m > self.omax:
                    self.frequent.pop(p2, 0)
                    pruned.add(p2)

    def lcs(self, p1: str, p2: str, n: int, m: int) -> int:
        """Longest common subsequence.

        Find the length of the longest sequence that is contained in two
        patterns. Uses dynamic programming algorithm adapted from
        https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/
        """
        L = [[0] * (m + 1) for i in range(n + 1)]
        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif p1[i - 1] == p2[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])
        return L[n][m]
