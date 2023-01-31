"""Motif miner module.

This module defines the mine_motifs function, which takes a database of
time series and finds frequent or maximal motifs in it. The motifs can
be filtered for length and ranked using different strategies.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .preprocessing import sax, breakpoints
from .PatternMiner import PatternMiner


class Miner:
    """Motif miner class.
    
    Parameters
    ----------
    timeseries : tensorflow.Tensor
        2D array with a collection of time series.
    min_sup : float
        Fraction of time series a motif should occur in.
    w : int
        Window size for Piecewise Aggregate Approximation.
    a : int
        Alphabet size for discretization.
    l : int
        Minimal motif length.
    o: float
        Maximal fraction of patterns contained in longer patters.
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
    motifs : list
        Constructed motifs ordered by the distances to their occurrences.
    """

    def __init__(
            self,
            timeseries: tf.Tensor,
            min_sup: float,
            w: int,
            a: int,
            l: int,
            o: float,
            k: int,
            m: bool
    ):
        self.timeseries = timeseries
        self.min_sup = min_sup
        self.w = w
        self.a = a
        self.l = l
        self.o = o
        self.k = k
        self.m = m

        self.sequences = None
        self.motifs = None

    def mine_motifs(self):
        """Perform all steps in motif mining pipeline."""
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
        pm = PatternMiner(self.sequences, self.min_sup)
        pm.mine()
        pm.prune_trivial(self.l, self.o)

        # Get indexes of frequent or maximal patterns
        if self.m:
            self.motifs = pm.maximal.values()
        else:
            self.motifs = pm.frequent.values()

    def map_patterns(self):
        """Map patterns back to motifs."""
        self.motifs = [self.map(motif) for motif in self.motifs]

        self.motifs.sort(key=lambda motif: motif.rmse)

    def map(self, motif):
        """Fill motif attributes."""
        motif.occurrences = self.get_occurrences(motif)
        motif.map()

        return motif

    def get_occurrences(self, motif):
        """Get the occurrences of one motif in the time series."""
        motif_len = len(motif.pattern) * self.w

        occurrences = {}
        for i, indexes in motif.indexes.items():
            occ = []
            for index in indexes:
                start = index * self.w
                end = start + motif_len
                occurrence = self.timeseries[i][start: end]

                # Ensure motif occurrences are indeed all the same length
                if too_short := motif_len - len(occurrence):
                    start -= too_short
                    occurrence = self.timeseries[i][start: end]

                occ.append(occurrence)

            occurrences[i] = occ

        return occurrences

    def plot(self, motif, kind='all'):
        if kind in ['all', 'representative']:
            self._plot_representative(motif)
        elif kind in ['all', 'timeseries']:
            self._plot_timeseries(motif)
        elif kind in ['all', 'box']:
            self._plot_box(motif)
        else:
            raise ValueError(f'Invalid value for `kind`: "{kind}"')

    def _plot_representative(self, motif):
        plt.plot(tf.transpose(motif.matches), 'k', lw=0.1)
        plt.plot(motif.representative, 'b', lw=1)
        plt.hlines(breakpoints[self.a], 0, len(motif.representative), 'k', lw=0.3)
        plt.ylim(-3, 3)
        plt.show()

    def _plot_timeseries(self, motif):
        for k, m in motif.match_indexes.items():
            x = list(range(len(self.timeseries[k])))
            y = self.timeseries[k]
            start, end = m[0] * self.w, m[1] * self.w
            plt.plot(x[:start + 1], y[:start + 1], 'k', lw=0.5)
            plt.plot(x[start:end + 1], y[start:end + 1], 'b', lw=1.5)
            plt.plot(x[start:], y[end:], 'k', lw=0.5)
        plt.show()

    def _plot_box(self, motif):
        fig, ax = plt.subplots(len(motif.match_indexes), sharex='all', sharey='all')
        for i, (k, m) in enumerate(motif.match_indexes.items()):
            # m[0], m[1] = m[0] * w, m[1] * w
            ax[i].plot(self.timeseries[k], 'k')
            ax[i].set_ylim((-3, 3))
            ax[i].add_patch(Rectangle((m[0], -3), m[1] - m[0], 6, alpha=0.3))
        plt.subplots_adjust(hspace=0)
        plt.show()
