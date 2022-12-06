"""Frequent motif module.

This module defines the Mapper class, a class that takes a
collection of timeseries, discretises it to sequences, mines frequent
patterns in the sequences and finds the frequent patterns occurences in
the timeseries.
"""
from math import ceil
from string import ascii_lowercase as lowercase
from collections import defaultdict

import numpy as np

from .motif import Motif

class Mapper:
    """Find frequent motifs of variable lengths in timeseries.

    Parameters
    ----------
    timeseries : numpy.ndarray
        Collection of time series to mine frequent motifs in.
    w : int
        Window size for Piecewise Aggregate Approximation; factor by
        which the sequences are shorter than the time series.
    """
    def __init__(self, timeseries: np.ndarray, w: int):
        self.timeseries = timeseries
        self._w = w

    def map(self, motif: str)-> np.ndarray:
        """Translate a pattern to a motif."""
        motif.occurences = self.get_occurences(motif)

        motif.representative = self.get_representative(motif)

        motif.matches, motif.rmse = self.get_matches(motif)

        return motif   

    def get_occurences(self, motif: Motif)-> list:
        """Find occurences of a pattern in the time series."""
        motif_len = len(motif.pattern) * self._w

        occ_by_ts = []
        for i, indexes in motif.indexes.items():
            occ_in_ts = []
            for index in indexes:
                start = index * self._w
                end = start + motif_len
                occ_in_ts.append(self.timeseries[i][start : end])
            occ_by_ts.append(occ_in_ts)
        
        return occ_by_ts

    def get_representative(self, motif):
        """Get representative motif from occurences."""
        return self._mean([self._mean(occ) for occ in motif.occurences])

    def get_matches(self, motif: Motif):
        """Get best matches and RMSE to representative motif."""
        matches = []
        rmse = 0.0

        for occ_in_ts in motif.occurences:
            best_match = None
            min_rmse = 100
            for occurence in occ_in_ts:
                rmse = np.mean((occurence - motif.representative) ** 2) ** 0.5
                if rmse < min_rmse:
                    best_match = occurence
                    min_rmse = rmse
            rmse += min_rmse
            matches.append(best_match)

        return matches, rmse

    def _mean(self, a: list):
        """Wrapper around np.array and np.mean(axis=0)."""
        return np.mean(np.array(a), axis=0)  