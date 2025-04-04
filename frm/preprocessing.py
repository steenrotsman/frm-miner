"""Preprocessing module.

This module defines two time series preprocessing functions for standardisation and SAX representation.
"""

import numpy as np
from scipy.stats import norm, zscore


def sax(timeseries, seglen, alpha):
    """Symbolic Aggregate approXimation.

    Parameters
    ----------
    timeseries : list
        List of lists containing normalised time series to dicretise.
    seglen : int
        Segment length for PAA, factor by which discrete representation is shorter than time series.
    alpha : int
        Alphabet size; number of discrete elements the time series are to be binned into.

    Returns
    -------
    List with a collection of discrete sequences from the time series.
    """
    breakpoints = get_breakpoints(alpha)
    return [get_sax(series, seglen, breakpoints) for series in timeseries]


def get_sax(series, seglen, breakpoints):
    """Get SAX representation of one time series."""
    # No paa step necessary when seglen=1
    if seglen == 1:
        paa = series
    else:
        if too_short := (len(series) % seglen):
            append = [np.mean(series[-too_short:])] * (seglen - too_short)
            series = np.append(series, append)

        segments = series.reshape((-1, seglen))
        paa = np.mean(segments, axis=1)

    discretised = np.digitize(paa, breakpoints) + ord("a")
    return "".join(chr(x) for x in discretised)


def standardise(timeseries):
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
        return np.nan_to_num(zscore(timeseries, axis=1))
    except ValueError:
        return [np.nan_to_num(zscore(ts)) for ts in timeseries]


def get_breakpoints(a):
    return norm.ppf(np.arange(1, a) / a, loc=0)
