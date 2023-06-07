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
    minsup : float
        Fraction of time series a motif should occur in.
    seglen : int
        Segment length for Piecewise Aggregate Approximation.
    alphabet : int
        Alphabet size for discretisation.
    min_len : int
        Minimal pattern length.
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
            minsup: float,
            seglen: int,
            alphabet: int,
            min_len: int,
            max_overlap: float,
            local: bool = True,
            k: int = 0,
            maximal: bool = False
    ):
        self.minsup = minsup
        self.seglen = seglen
        self.alphabet = alphabet
        self.min_len = min_len
        self.max_overlap = max_overlap
        self.local = local
        self.k = k
        self.maximal = maximal

        self.sequences = None
        self.motifs = None

    def mine_motifs(self, ts):
        """Perform all steps in motif mining pipeline."""
        st = standardise(ts, self.local)
        ds = sax(st, self.seglen, self.alphabet)
        self.mine_patterns(ds)
        self.map_patterns(ts)
        self.sort_patterns()

        return self.motifs if not self.k else self.motifs[:self.k]

    def mine_patterns(self, ds):
        """Find frequent patterns in the sequences."""
        # Find frequent and maximal patterns in the sequences
        pm = PatternMiner(self.minsup, self.min_len, self.max_overlap)
        pm.mine(ds)

        # Get indexes of frequent or maximal patterns
        if self.maximal:
            self.motifs = list(pm.maximal.values())
        else:
            self.motifs = list(pm.frequent.values())

    def map_patterns(self, ts):
        """Map patterns back to motifs."""
        for motif in self.motifs:
            motif.occurrences = self.get_occurrences(motif, ts)
            motif.map()
            motif.match_indexes = {k: [s * self.seglen, e * self.seglen] for k, (s, e) in motif.match_indexes.items()}

    def sort_patterns(self):
        """Sort patterns on their root mean squared error of representative and occurrences."""
        self.motifs.sort(key=lambda motif: motif.rmse)

    def get_occurrences(self, motif, ts):
        """Get the occurrences of one motif in the time series."""
        motif_len = len(motif.pattern) * self.seglen

        occurrences = {}
        for i, indexes in motif.indexes.items():
            occ = []
            for index in indexes:
                start = index * self.seglen
                end = start + motif_len
                occurrence = ts[i][start: end]

                # Ensure motif occurrences are indeed all the same length
                if too_short := motif_len - len(occurrence):
                    start -= too_short
                    occurrence = ts[i][start: end]

                occ.append(occurrence)

            occurrences[i] = occ

        return occurrences
