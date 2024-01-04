"""Preprocessing module.

This module defines two time series preprocessing functions for standardisation and SAX representation.
"""
import numpy as np
from scipy.stats import zscore, norm


def sax(timeseries, seglen, alphabet):
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


def get_sax(series, seglen, breakpoints):
    """Get SAX representation of one time series."""
    segments = [sum(series[i:i + seglen]) / seglen for i in range(0, len(series), seglen)]
    for i, seg in enumerate(segments):
        for b, c in breakpoints[a].items():
            if seg <= b:
                segments[i] = c
                break
        else:
            segments[i] = chr(ord(c) + 1)

    sax_representation = ''.join(segments)
    return sax_representation



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
