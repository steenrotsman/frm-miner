"""Motif miner module.

This module defines the mine_motifs function, which takes a database of
time series and finds frequent or maximal motifs in it. The motifs can
be filtered for length and ranked using different strategies.
"""

from operator import attrgetter

from .patterns import PatternMiner
from .preprocessing import sax, standardise


class Miner:
    """Motif miner class.

    Parameters
    ----------
    minsup : float
        Fraction of time series a motif should occur in.
    seglen : int
        Segment length for Piecewise Aggregate Approximation.
    alpha : int
        Alphabet size for discretisation.
    omax: float, optional
        Maximal fraction of patterns contained in longer patters.
    mass : bool, optional
        Whether to scan the time series database to look for additional matches using MASS.
    p : int
        If 2, ranks motifs on ED (FRM-Miner 2.0). If 1, uses NAED (FRM-Miner 1.0)
    k : int, optional
        Number of motifs to return. If 0, all motifs are returned.

    Attributes
    ----------
    motifs : list
        Constructed motifs ordered by the distances to their occurrences.
    """

    def __init__(self, minsup, seglen, alpha, omax=0.8, mass=False, k=0):
        self.minsup = minsup
        self.seglen = seglen
        self.alpha = alpha
        self.omax = omax
        self.mass = mass
        self.k = k

        self.motifs = None

    def mine(self, ts):
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
        discretised = sax(standardised, self.seglen, self.alpha)
        self.mine_patterns(discretised)
        self.map_patterns(standardised)

        return self.motifs if not self.k else self.motifs[: self.k]

    def mine_patterns(self, ds):
        """Find frequent patterns in the sequences.

        Parameters
        ----------
        sequences : list
            Collection of time series discretised to sequences.
        """
        pm = PatternMiner(self.minsup, self.omax)
        pm.mine(ds)
        self.motifs = list(pm.frequent.values())

    def map_patterns(self, ts):
        """Map patterns back to motifs."""
        for motif in self.motifs:
            motif.map(ts, self.seglen)
        self.motifs.sort(key=attrgetter('distance'))
        if self.mass:
            for motif in self.motifs[: self.k]:
                motif.get_more_matches()
