"""Motif miner module.

This module defines the mine_motifs function, which takes a database of
time series and finds frequent or maximal motifs in it. The motifs can
be filtered for length and ranked using different strategies.
"""
from .preprocessing import standardise, sax
from .patterns import PatternMiner


class Miner:
    """Motif miner class.
    
    Parameters
    ----------
    timeseries : list
        List of lists with akm database of time series.
    min_sup : float
        Fraction of time series a motif should occur in.
    seglen : int
        Segment length for Piecewise Aggregate Approximation.
    alphabet : int
        Alphabet size for discretisation.
    min_len : int
        Minimal motif length.
    max_overlap: float
        Maximal fraction of patterns contained in longer patters.
    local : bool
        Use the local (True) or global (False) mean and standard deviation for standardisation.
    k : int, optional
        Number of motifs to return.
        If 0, all motifs are returned.
    maximal : bool
        Return only maximal patterns. 
        If False, returns all frequent motifs.
    
    Attributes
    ----------
    sequences : list
        Collection of time series discretised to sequences.
    motifs : list
        Constructed motifs ordered by the distances to their occurrences.
    """

    def __init__(
            self,
            timeseries: list,
            min_sup: float,
            seglen: int,
            alphabet: int,
            min_len: int,
            max_overlap: float,
            local: bool = True,
            k: int = 0,
            maximal: bool = False
    ):
        self.timeseries = timeseries
        self.min_sup = min_sup
        self.seglen = seglen
        self.alphabet = alphabet
        self.min_len = min_len
        self.max_overlap = max_overlap
        self.local = local
        self.k = k
        self.maximal = maximal

        self.sequences = None
        self.motifs = None

    def mine_motifs(self):
        """Perform all steps in motif mining pipeline."""
        self.standardise()
        self.discretise()
        self.mine_patterns()
        self.map_patterns()
        self.sort_patterns()

        return self.motifs if not self.k else self.motifs[:self.k]

    def standardise(self):
        """Standardize time series to """
        self.timeseries = standardise(self.timeseries, self.local)

    def discretise(self):
        """Discretise time series to sequences."""
        self.sequences = sax(self.timeseries, self.seglen, self.alphabet)

    def mine_patterns(self):
        """Find frequent patterns in the sequences."""
        # Find frequent and maximal patterns in the sequences
        pm = PatternMiner(self.sequences, self.min_sup)
        pm.mine()
        pm.prune_trivial(self.min_len, self.max_overlap)

        # Get indexes of frequent or maximal patterns
        if self.maximal:
            self.motifs = pm.maximal.values()
        else:
            self.motifs = pm.frequent.values()

    def map_patterns(self):
        """Map patterns back to motifs."""
        self.motifs = [self.map(motif) for motif in self.motifs]

    def sort_patterns(self):
        """Sort patterns on their root mean squared error of representative and occurrences."""
        self.motifs.sort(key=lambda motif: motif.rmse)

    def map(self, motif):
        """Fill motif attributes."""
        motif.occurrences = self.get_occurrences(motif)
        motif.map()
        motif.match_indexes = {k: [s * self.seglen, e * self.seglen] for k, (s, e) in motif.match_indexes.items()}

        return motif

    def get_occurrences(self, motif):
        """Get the occurrences of one motif in the time series."""
        motif_len = len(motif.pattern) * self.seglen

        occurrences = {}
        for i, indexes in motif.indexes.items():
            occ = []
            for index in indexes:
                start = index * self.seglen
                end = start + motif_len
                occurrence = self.timeseries[i][start: end]

                # Ensure motif occurrences are indeed all the same length
                if too_short := motif_len - len(occurrence):
                    start -= too_short
                    occurrence = self.timeseries[i][start: end]

                occ.append(occurrence)

            occurrences[i] = occ

        return occurrences
