"""Provide sequence motif mining classes for profiling FRM-Miner 1.0 and 2.0 memory use.

This module defines two functions for measuring the size of the index map/tree for both
FRM-Miner 1.0 and FRM-Miner 2.0. The pympler.asizeof.asizeof function is used to measure
the size of the dictionary that defines the index map/tree. The run times are severely 
impacted by the memory profiling, especially for FRM-Miner 1.0.
"""

from pympler.asizeof import asizeof

from frm.motif import Motif
from frm.patterns import PatternMiner
from frm.preprocessing import sax, standardise


def profile_memory_peak(data, frm, minsup=0.3, seglen=2, alphabet=4):
    standardised = standardise(data)
    discretised = sax(standardised, seglen, alphabet)
    if frm == 1:
        pm = PatternMiner_1(minsup)
    elif frm == 2:
        pm = PatternMiner_2(minsup)
    else:
        raise ValueError(f"Can only profile FRM-Miner (1).0 and (2).0, got: {frm}")
    return pm.mine(discretised)


class PatternMiner_2(PatternMiner):
    def mine(self, sequences):
        self._min_freq = len(sequences) * self.minsup
        self.mine_1_patterns(sequences)
        peak = asizeof(self.frequent)
        self.prune_infrequent()

        while self._patterns[1] and (not self.max_len or self._k <= self.max_len):
            self._patterns = [self._patterns[1], set()]
            self.generate_candidates_from_parents(sequences)
            if size := asizeof(self.frequent):
                peak = size
            self.prune_infrequent()
            self._k += 1
        return peak

    def mine_1_patterns(self, sequences):
        for i, sequence in enumerate(sequences):
            for j, item in enumerate(sequence):
                if item not in self.frequent:
                    self.frequent[item] = Motif(item)
                    self._patterns[1].add(item)
                self.frequent[item].record_index(i, j)


class PatternMiner_1(PatternMiner):
    def mine(self, sequences):
        self._patterns = [[], []]
        self._min_freq = len(sequences) * self.minsup
        self.mine_1_patterns(sequences)
        peak = asizeof(self.frequent)
        for a in list(self.frequent.keys()):
            self.prune_infrequent(a)

        while self._patterns[self._k - 1]:
            self._patterns.append([])

            for candidate in self.get_candidates():
                self.frequent[candidate] = Motif(candidate)

                for seq, indexes in self.frequent[candidate[:-1]].indexes.items():
                    sequence = sequences[seq]
                    for index in indexes:
                        if sequence[index : index + self._k] == candidate:
                            self.frequent[candidate].record_index(seq, index)
                self.prune_infrequent(candidate)
            if size := asizeof(self.frequent):
                peak = size
            self._k += 1
        return peak

    def get_candidates(self):
        patterns = self._patterns[self._k - 1]
        return [p1 + p2[-1] for p1 in patterns for p2 in patterns if p1[1:] == p2[:-1]]

    def mine_1_patterns(self, sequences):
        for i, sequence in enumerate(sequences):
            for j, item in enumerate(sequence):
                if item not in self.frequent:
                    self.frequent[item] = Motif(item)
                self.frequent[item].record_index(i, j)

    def prune_infrequent(self, pattern):
        if len(self.frequent[pattern].indexes) < self._min_freq:
            self.frequent.pop(pattern)
        else:
            self._patterns[len(pattern)].append(pattern)
