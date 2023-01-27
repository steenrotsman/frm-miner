"""Preprocessing module.

This module defines various time series preprocessing functions, such as
normalization, dimensionality reduction and discretization.
"""
import tensorflow as tf


def sax(ts: tf.Tensor, w: int, a: int):
    """Symbolic Aggregate approXimation.
    
    Parameters
    ----------
    ts : tf.Tensor
        Potentially data 2D tensor of time series to dicretize.
    w : int
        Window size for PAA.
    a : int
        Alphabet size; number of discrete elements the time series are
        to be binned into.
    
    Returns
    -------
    List with a collection of discrete sequences from the time series.
    """
    def _discretize(row):
        discrete = tf.searchsorted(breakpoints[a], row)
        discrete += 97
        discrete = tf.strings.unicode_encode(discrete, 'UTF-8')
        return discrete.numpy().decode('utf-8')

    standardized = standardize(ts)
    aggregated = paa(standardized, w)
    return [_discretize(row) for row in aggregated]


def standardize(ts: tf.Tensor):
    """Standardize time series to a mean of 0 and a std of 1."""
    avg = tf.math.reduce_mean(ts)
    std = tf.math.reduce_std(ts)
    return (ts - avg) / std


def paa(ts: tf.Tensor, w: int):
    """Perform Piecewise Aggregate Approximation."""
    if w == 1:
        return ts

    def _paa(row):
        series = []
        j = 0
        while len(series) * w < tf.shape(row):
            series.append(tf.math.reduce_mean(row[j*w:(j+1)*w]))
            j += 1
        return tf.stack(series)

    # Process row by row to accommodate data tensors
    return tf.map_fn(_paa, ts)


# Defined in Lin, Keogh, Linardi, & Chiu (2003). A Symbolic Representation of Time Series,
# with Implications for Streaming Algorithms. Table 3.
breakpoints = {
    3: [-0.43, 0.43],
    4: [-0.67, 0, 0.67],
    5: [-0.84, -0.25, 0.25, 0.84],
    6: [-0.97, -0.43, 0, 0.43, 0.97],
    7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
    8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
    9: [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
    10: [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
}