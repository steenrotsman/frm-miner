"""Preprocessing module.

This module defines various time series preprocessing functions, such as
normalization, dimensionality reduction and discretization.
"""
import numpy as np

def sax(ts: np.ndarray, w: int, a: int):
    """Symbolic Aggregate approXimation.
    
    Parameters
    ----------
    ts : numpy.ndarray
        2D array of time series to dicretize.
    w : int
        Window size for PAA.
    a : int
        Alphabet size; number of discrete elements the time series are
        to be binned into.
    
    Returns
    -------
    List with a collection of discrete sequences from the time series.
    """
    sax = _sax(ts, w, a)
    dim = len(sax.shape)
    
    if dim == 1:
        return _to_string(sax)
    elif dim == 2:
        return [_to_string(sequence) for sequence in sax]


def _sax(ts: np.ndarray, w: int, a: int):
    """Fast SAX implementation."""

    # Defined in Lin, Keogh, Lonardi, & Chiu (2003). A Symbolic
    # Representation of Time Series, with Implications for Streaming
    # Algorithms. Table 3.
    breakpoints = {
        3: np.array([-0.43, 0.43]),
        4: np.array([-0.67, 0, 0.67]),
        5: np.array([-0.84, -0.25, 0.25, 0.84]),
        6: np.array([-0.97, -0.43, 0, 0.43, 0.97]),
        7: np.array([-1.07, -0.57, -0.18, 0.18, 0.57, 1.07]),
        8: np.array([-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15]),
        9: np.array([-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22]),
        10: np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28]),
    }

    return np.digitize(_paa(_standardize(ts), w), breakpoints[a])


def _standardize(ts: np.ndarray):
    """Standardize time series to a mean of 0 and a std of 1."""
    avg = np.mean(ts)
    std = np.std(ts)
    return (ts - avg) / std


def _paa(ts: np.ndarray, w: int):
    """Perform Piecewise Aggregate Approximation."""
    dim = len(ts.shape)
    
    if dim == 1:
        new_length = int(ts.shape[0] / w)
        paa = np.empty((new_length))
        for i in range(new_length):
            paa[i] = np.mean(ts[i*w : (i+1)*w])
    elif dim == 2:
        new_length = int(ts.shape[1] / w)
        paa = np.empty((ts.shape[0], new_length))
        for i in range(ts.shape[0]):
            serie = ts[i]
            for j in range(new_length):
                paa[i, j] = np.mean(serie[j*w:(j+1)*w])

    return paa


def _to_string(sequence: np.ndarray):
    """Transform sequence of integers to string of letters."""
    return ''.join([chr(c + 97) for c in sequence])
