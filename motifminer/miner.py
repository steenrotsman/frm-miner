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
    """Motif miner class."""
    def __init__(
        self,
        data: np.ndarray,
        w: int,
        a: int,
        l: int,
        topk: int = 0,
        maximal: bool = True):
            self.data = data
            self.w = w
            self.a = a
            self.l = l
            self.topk = topk
            self.maximal = maximal

    def mine_motifs(self):
        """Mine motifs from a database of time series.

        Parameters
        ----------
        data : numpy.ndarray
            2D array with a collection of time series.
        w : int
            Window size for Piecewise Aggregate Appriximation.
        a : int
            Alphabet size for discretization.
        l : int
            Minimum motif length.
        topk : int, optional
            Number of motifs to return.
            If 0 (default), all motifs are returned.
        maximal : bool
            Return only maximal patterns (default).
            If False, returns all frequent motifs.

        Returns
        -------
        Numpy array of time series occurences.
        At most one occurence is selected for each time series.
        """
        # {pattern: {idx of time series pattern occurs in: [start_idx]}}
        indexes = self.get_indexes()

        # [np.ndarray(motif)]
        mm = Mapper(self.data, indexes, self.w)
        motifs = [mm.map(pattern) for pattern in indexes]

        motifs.sort(key=lambda x: x[1])
        return motifs

    def get_indexes(self):
        # Discretize time series to sequences
        sequences = sax(self.data, self.w, self.a)

        # Find frequent and maximal patterns in the sequences
        gsp = GSP(sequences, self.a)
        gsp.mine()
        gsp.prune(self.l)

        # Get indexes of frequent or maximal patterns
        if self.maximal:
            return gsp.maximal
        return gsp.frequent
            