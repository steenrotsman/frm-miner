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
    minsup : float
        Fraction of time series a motif should occur in.
    seglen : int
        Segment length for Piecewise Aggregate Approximation.
    alphabet : int
        Alphabet size for discretisation.
    min_len : int
        Minimal pattern length.
    max_len : int
        Maximal pattern length.
        If 0, limit is infinity.
    max_overlap: float
        Maximal fraction of patterns contained in longer patters.
    k : int, optional
        Number of motifs to return.
        If 0, all motifs are returned.
    maximal : bool
        Return only maximal patterns. 
        If False, returns all frequent motifs.
    
    Attributes
    ----------
    motifs : list
        Constructed motifs ordered by the distances to their occurrences.
    """
    def __init__(
            self,
            minsup: float,
            seglen: int,
            alphabet: int,
            min_len: int = 3,
            max_len: int = 0,
            max_overlap: float = 0.9,
            k: int = 0,
    ):
        self.minsup = minsup
        self.seglen = seglen
        self.alphabet = alphabet
        self.min_len = min_len
        self.max_len = max_len
        self.max_overlap = max_overlap
        self.k = k

        self.motifs = None

    def mine_motifs(self, ts):
        """Perform all steps in motif mining pipeline.

        Parameters
        ----------
        ts : list
            Database of time series.

        Returns
        -------
        res: list
            frequent motifs.
        """
        standardised = standardise(ts)
        discretised = sax(standardised, self.seglen, self.alphabet)
        self.mine_patterns(discretised)
        self.map_patterns(standardised)
        self.sort_patterns()

        return self.motifs if not self.k else self.motifs[:self.k]

    def mine_patterns(self, ds):
        """Find frequent patterns in the sequences.

        Parameters
        ----------
        sequences : list
            Collection of time series discretised to sequences.
        """
        pm = PatternMiner(self.minsup, self.min_len, self.max_len, self.max_overlap)
        pm.mine(ds)
        self.motifs = list(pm.frequent.values())

    def map_patterns(self, ts):
        """Map patterns back to motifs."""
        for motif in self.motifs:
            motif.occurrences = self.get_occurrences(motif, ts)
            motif.map()
            motif.match_indexes = {k: [s * self.seglen, e * self.seglen] for k, (s, e) in motif.match_indexes.items()}

    def sort_patterns(self):
        """Sort patterns on their root mean squared error of representative and occurrences."""
        self.motifs.sort(key=lambda motif: motif.naed)

    def get_occurrences(self, motif, ts):
        """Get the occurrences of one motif in the time series.

        Parameters
        ----------
        motif : Motif
            Motif object to find occurrences from.
        ts : list
            Time series to find occurrences in.

        Returns
        -------
        occurrences : dict
            Dictionary with time series indexes as keys and matching
            subsequences as values.
        """
        motif_len = len(motif.pattern) * self.seglen

        occurrences = {}
        for i, indexes in motif.indexes.items():
            occ = []
            for index in indexes:
                start = index * self.seglen
                end = start + motif_len

                # Ensure motif occurrences are all the same length
                if too_short := max(0, end - len(ts[i])):
                    start -= too_short

                occ.append(ts[i][start: end])

            occurrences[i] = occ

        return occurrences
