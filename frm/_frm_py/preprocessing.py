"""Preprocessing module.

This module defines two time series preprocessing functions for standardisation and SAX representation.
"""
from typing import Union

from scipy.stats import zscore
from numpy import nan_to_num

# https://stackoverflow.com/a/60617044
numeric = Union[int, float]


def sax(timeseries: list[list[numeric]], seglen: int, alphabet: int) -> list[str]:
    """Symbolic Aggregate approXimation.
    
    Parameters
    ----------
    timeseries : list
        List of lists containing normalised time series to dicretise.
    seglen : int
        Segment length for PAA, factor by which discrete representation is shorter than time series.
    alphabet : int
        Alphabet size; number of discrete elements the time series are to be binned into.
    
    Returns
    -------
    List with a collection of discrete sequences from the time series.
    """
    return [get_sax(series, seglen, alphabet) for series in timeseries]


def get_sax(series: list[numeric], seglen: int, a: int) -> str:
    """Get SAX representation of one time series."""
    segments = [sum(series[i:i + seglen]) / seglen for i in range(0, len(series) - seglen + 1, seglen)]
    for i, seg in enumerate(segments):
        for b, c in breakpoints[a].items():
            if seg <= b:
                segments[i] = c
                break
        else:
            segments[i] = chr(ord(c) + 1)

    sax_representation = ''.join(segments)
    return sax_representation


def standardise(timeseries: list[list]):
    """Standardise time series.

    Standardises data to have a mean of zero and a standard deviation of one.

    Parameters
    ----------
    timeseries
        Database of time series to standardise.

    Returns
    -------
    standardised
        Database of standardised time series.
    """
    try:
        return nan_to_num(zscore(timeseries, axis=1))
    except ValueError:
        return [nan_to_num(zscore(ts)) for ts in timeseries]


# Defined in Lin, Keogh, Linardi, & Chiu (2003). A Symbolic Representation of Time Series,
# with Implications for Streaming Algorithms. Table 3.
breakpoints = {
    2: {0: 'a'},
    3: {-0.43: 'a', 0.43: 'b'},
    4: {-0.67: 'a', 0: 'b', 0.67: 'c'},
    5: {-0.84: 'a', -0.25: 'b', 0.25: 'c', 0.84: 'd'},
    6: {-0.97: 'a', -0.43: 'b', 0: 'c', 0.43: 'd', 0.97: 'e'},
    7: {-1.07: 'a', -0.57: 'b', -0.18: 'c', 0.18: 'd', 0.57: 'e', 1.07: 'f'},
    8: {-1.15: 'a', -0.67: 'b', -0.32: 'c', 0: 'd', 0.32: 'e', 0.67: 'f', 1.15: 'g'},
    9: {-1.22: 'a', -0.76: 'b', -0.43: 'c', -0.14: 'd', 0.14: 'e', 0.43: 'f', 0.76: 'g', 1.22: 'h'},
    10: {-1.28: 'a', -0.84: 'b', -0.52: 'c', -0.25: 'd', 0: 'e', 0.25: 'f', 0.52: 'g', 0.84: 'h', 1.28: 'i'},
}
