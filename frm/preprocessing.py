"""Preprocessing module.

This module defines two time series preprocessing functions for standardisation and SAX representation.
"""
import itertools
from typing import Union

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
    sax_representation = ''.join(segments)
    return sax_representation


def standardise(timeseries: list[list], local: bool = True) -> list[list[numeric]]:
    """Standardise time series.

    Standardises data to have a mean of zero and a standard deviation of one.

    Parameters
    ----------
    timeseries
        Database of time series to standardise.
    local : bool
        Use the local (True) or global (False) mean and standard deviation for standardisation.

    Returns
    -------
    standardised
        Database of standardised time series.
    """
    if local:
        standardised = []
        for series in timeseries:
            n = len(series)
            mean = sum(series) / n
            standardised.append(get_standardised(series, mean, get_std(series, mean, n)))
    else:
        flat_ts = list(itertools.chain.from_iterable(timeseries))
        n = len(flat_ts)
        global_mean = sum(flat_ts) / n
        global_std = get_std(flat_ts, global_mean, n)
        standardised = [get_standardised(series, global_mean, global_std) for series in timeseries]

    return standardised


def get_standardised(series: list[numeric], m: float, std: float) -> list[float]:
    """Perform standardisation on one time series."""
    return [(x - m) / std for x in series]


def get_std(series: list[numeric], mean: float, n: int) -> float:
    """Calculate the standard deviation of a list given its mean and length."""
    return (sum((x - mean) ** 2 for x in series) / n) ** 0.5


# Defined in Lin, Keogh, Linardi, & Chiu (2003). A Symbolic Representation of Time Series,
# with Implications for Streaming Algorithms. Table 3.
breakpoints = {
    3: {-0.43: 'a', 0.43: 'b', 10: 'c'},
    4: {-0.67: 'a', 0: 'b', 0.67: 'c', 10: 'd'},
    5: {-0.84: 'a', -0.25: 'b', 0.25: 'c', 0.84: 'd', 10: 'e'},
    6: {-0.97: 'a', -0.43: 'b', 0: 'c', 0.43: 'd', 0.97: 'e', 10: 'f'},
    7: {-1.07: 'a', -0.57: 'b', -0.18: 'c', 0.18: 'd', 0.57: 'e', 1.07: 'f', 10: 'g'},
    8: {-1.15: 'a', -0.67: 'b', -0.32: 'c', 0: 'd', 0.32: 'e', 0.67: 'f', 1.15: 'g', 10: 'h'},
    9: {-1.22: 'a', -0.76: 'b', -0.43: 'c', -0.14: 'd', 0.14: 'e', 0.43: 'f', 0.76: 'g', 1.22: 'h', 10: 'i'},
    10: {-1.28: 'a', -0.84: 'b', -0.52: 'c', -0.25: 'd', 0: 'e', 0.25: 'f', 0.52: 'g', 0.84: 'h', 1.28: 'i', 10: 'j'},
}
