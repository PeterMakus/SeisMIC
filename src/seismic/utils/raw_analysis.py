'''
Module for waveform data analysis. Contains spectrogram computation.

:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 21st June 2023 12:22:00 pm
Last Modified: Wednesday, 19th July 2023 11:55:05 am
'''
from typing import Iterator
import warnings

import numpy as np
from obspy import Stream, Trace
from scipy.signal import welch
from scipy.interpolate import pchip_interpolate

from seismic.utils.miic_utils import resample_or_decimate


def spct_series_welch(
        streams: Iterator[Stream], window_length: int, freqmax: float):
    """
    Computes a spectral time series. Each point in time is computed using the
    welch method. Windows overlap by half the windolength. The input stream can
    contain one or several traces from the same station. Frequency axis is
    logarithmic.
    :param st: Input Stream with data from one station.
    :type st: ~obspy.core.Stream
    :param window_length: window length in seconds for each datapoint in time
    :type window_length: int or float
    :param freqmax: maximum frequency to be considered
    :type freqmax: float
    :return: Arrays containing a frequency and time series and the spectral
        series.
    :rtype: np.ndarray
    """
    specl = []
    # List of actually available times
    t = []
    for st in streams:
        try:
            # Don't preprocess masked values
            tr = Stream(
                [preprocess(tr, freqmax) for tr in st.split()]).merge()[0]
        except IndexError:
            warnings.warn('No data in stream for this time step.')
            continue
        for wintr in tr.slide(window_length=window_length, step=window_length):
            f, S = welch(wintr.data, fs=wintr.stats.sampling_rate)
            # interpolate onto a logarithmic frequency space
            # 512 points of resolution in f direction hardcoded for now
            f2 = np.logspace(-3, np.log10(f.max()), 512)
            S2 = pchip_interpolate(f, S, f2)
            specl.append(S2)
            t.append(wintr.stats.starttime)
    S = np.array(specl)

    t = np.array(t)
    return f2, t, S.T


def preprocess(tr: Trace, freqmax: float):
    """
    Some very basic preprocessing on the string in order to plot the spectral
    series. Does the following steps:
    *1. Remove station response*
    *2. Detrend*
    *3. Decimate if sampling rate>50*
    *4. highpass filter with corner period of 300s.*
    :param st: The input Stream, should only contain Traces from one station.
    :type st: ~obspy.core.Stream
    :return: The output stream and station inventory object
    :rtype: ~obspy.core.Stream and ~obspy.core.Inventory
    """
    # Downsample to make computations faster
    resample_or_decimate(tr, freqmax*2)
    tr.remove_response()
    # Detrend
    tr.detrend(type='linear')

    # highpass filter
    tr.filter('highpass', freq=0.01)

    return tr
