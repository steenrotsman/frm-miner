"""Motif miner module.

This module defines the mine_motifs function, which takes a database of
time series and finds frequent or maximal motifs in it. The motifs can
be filtered for length and ranked using different strategies.
"""
from typing import Callable

import numpy as np

from .preprocessing import sax
from .GSP import GSP
from .mapper import Mapper

class Miner:
    """Motif miner class.
    
    Parameters
    ----------
    timeseries : numpy.ndarray
        2D array with a collection of time series.
    min_sup : float
        Fraction of time series a motif should occur in.
    w : int
        Window size for Piecewise Aggregate Appriximation.
    a : int
        Alphabet size for discretization.
    l : int
        Minimum motif length.
    top_k : int, optional
        Number of motifs to return.
        If 0 (default), all motifs are returned.
    maximal : bool
        Return only maximal patterns (default).
        If False, returns all frequent motifs.
    
    Attributes
    ----------
    sequences : Iterable
        Collection of time series discretized to sequences.
    index_map : dict
        Nested dictionary that maps frequent or maximal patterns to
        indexes of pattern occurences.
    motifs : list
        List of constructed motifs ordered by the distances to their
        occurences.
    """
    def __init__(
        self,
        timeseries: np.ndarray,
        min_sup: float,
        w: int,
        a: int,
        l: int,
        top_k: int = 0,
        maximal: bool = True):
            self.timeseries = timeseries
            self.min_sup = min_sup
            self.w = w
            self.a = a
            self.l = l
            self.top_k = top_k
            self.maximal = maximal

    def mine_motifs(self):
        """Perform all steps in motifminer pipeline."""
        self.discretize()
        self.mine_patterns()
        self.map_patterns()

        return self.motifs if not self.top_k else self.motifs[:self.top_k]

    def discretize(self):
        """Discretize time series to sequences."""
        self.sequences = sax(self.timeseries, self.w, self.a)

    def mine_patterns(self):
        """Find frequent patterns in the sequences."""
        # Find frequent and maximal patterns in the sequences
        gsp = GSP(self.sequences, self.a)
        gsp.mine()
        gsp.prune(self.l)

        # Get indexes of frequent or maximal patterns
        if self.maximal:
            self.index_map = gsp.maximal
        else:
            self.index_map = gsp.frequent

    def map_patterns(self):
        """Map patterns back to motifs."""
        mm = Mapper(self.timeseries, index_map, self.w)
        self.motifs = [mm.map(pattern) for pattern in index_map]

        self.motifs.sort(key=lambda x: x[1])
