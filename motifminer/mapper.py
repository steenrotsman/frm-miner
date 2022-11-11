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

class Mapper:
    """Find frequent motifs of variable lengths in timeseries.

    Parameters
    ----------
    timeseries : numpy.ndarray
        Collection of time series to mine frequent motifs in.
    indexes : dict
        See GSP.indexes.
    w : int
        Window size for Piecewise Aggregate Approximation; factor by
        which the sequences are shorter than the time series.
    """
    def __init__(self, timeseries: np.ndarray, indexes: dict, w: int):
        self.timeseries = timeseries
        self.indexes = indexes
        self._w = w

    def map(self, pattern: str)-> np.ndarray:
        """Translate a pattern to a motif.

        Maps all occurences of the pattern to the collection of time
        series and constructs a synthetic representative motif. This
        representative motif can then be used directly, or to find the
        most representative motif occurence.

        Parameters
        ----------
        pattern : str
            The pattern to map to the time series.
        
        Returns
        -------
        motif : np.ndarray
            A constructed average motif from all occurences.
        rmse : float
            Average RMSE between the synthetic motif and its occurences.
        """
        motifslist = self._find(pattern)
        synthetic = self._mean([self._mean(motifs) for motifs in motifslist])

        rmse = 0
        for motifs in motifslist:
            rmse += min([self._rmse(motif, synthetic) for motif in motifs])
        
        return synthetic, rmse / len(motifslist)    

    def _find(self, pattern: str)-> dict:
        """Find occurences of a pattern in the time series.

        Uses the indexes from miner to find the occurences of a pattern
        in all timeseries.

        Parameters
        ----------
        pattern : str
            The pattern to look for in the timeseries.

        Returns
        -------
        List of lists with motif occurences in the time series.
        Occurences from the same time series are grouped together in the
        outer list.
        """
        motif_len = len(pattern) * self._w

        occurences = []
        for i, indexes in self.indexes[pattern].items():
            motifs = []
            for index in indexes:
                start = index * self._w
                end = start + motif_len
                motifs.append(self.timeseries[i][start : end])
            occurences.append(motifs)
        
        return occurences

    def _mean(self, a: list):
        """Wrapper around np.array and np.mean(axis=0)."""
        return np.mean(np.array(a), axis=0)
    
    def _rmse(self, a: np.ndarray, b: np.ndarray):
        """Root Mean Squared Error."""
        return np.mean((a - b) ** 2) ** 0.5    