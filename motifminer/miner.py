"""Motif miner module.

This module defines the mine_motifs function, which takes a database of
time series and finds frequent or maximal motifs in it. The motifs can
be filtered for length and ranked using different strategies.
"""
import tensorflow as tf

from .preprocessing import sax
from .GSP import GSP

class Miner:
    """Motif miner class.
    
    Parameters
    ----------
    timeseries : tensorflow.Tensor
        2D array with a collection of time series.
    min_sup : float
        Fraction of time series a motif should occur in.
    w : int
        Window size for Piecewise Aggregate Appriximation.
    a : int
        Alphabet size for discretization.
    l : int
        Minimum motif length.
    k : int, optional
        Number of motifs to return.
        If 0, all motifs are returned.
    m : bool
        Return only maximal patterns. 
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
        timeseries: tf.Tensor,
        min_sup: float,
        w: int,
        a: int,
        l: int,
        k: int,
        m: bool
        ):
            self.timeseries = timeseries
            self.min_sup = min_sup
            self.w = w
            self.a = a
            self.l = l
            self.k = k
            self.m = m

    def mine_motifs(self):
        """Perform all steps in motifminer pipeline."""
        self.discretize()
        self.mine_patterns()
        self.map_patterns()

        return self.motifs if not self.k else self.motifs[:self.k]

    def discretize(self):
        """Discretize time series to sequences."""
        self.sequences = sax(self.timeseries, self.w, self.a)

    def mine_patterns(self):
        """Find frequent patterns in the sequences."""
        # Find frequent and maximal patterns in the sequences
        gsp = GSP(self.sequences, self.min_sup)
        gsp.mine()
        gsp.prune(self.l)

        # Get indexes of frequent or maximal patterns
        if self.m:
            self.motifs = gsp.maximal.values()
        else:
            self.motifs = gsp.frequent.values()

    def map_patterns(self):
        """Map patterns back to motifs."""
        self.motifs = [self.map(motif) for motif in self.motifs]

        self.motifs.sort(key=lambda motif: motif.rmse)
    
    def map(self, motif):
        """Fill motif attributes."""
        motif.occurences = self.get_occurences(motif)
        motif.map()

        return motif

    def get_occurences(self, motif):
        """Get the occurences of one motif in the time series."""
        motif_len = len(motif.pattern) * self.w

        occurences = {}
        for i, indexes in motif.indexes.items():
            occ = []
            for index in indexes:
                start = index * self.w
                end = start + motif_len
                occurence = self.timeseries[i][start : end]

                # Ensure motif occurences are indeed all the same length
                if too_short := motif_len - len(occurence):
                    start -= too_short
                    occurence = self.timeseries[i][start : end]
 
                occ.append(occurence)
 
            occurences[i] = occ
        
        return occurences