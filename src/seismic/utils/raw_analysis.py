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
Last Modified: Wednesday, 21st June 2023 01:03:23 pm
'''
from typing import Tuple, List, Generator
from datetime import datetime
import locale
import os
import warnings
import logging

import numpy as np
from obspy import UTCDateTime, Stream, read_inventory, Trace, read
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.colors as colors
from scipy.signal import welch
from scipy.interpolate import pchip_interpolate

from seismic.utils.miic_utils import resample_or_decimate

# Directory structure
file = '{date.year}/{network}/{station}/{channel}.D/{network}.{station}.*.{channel}.D.{date.year}.{doy}'

# Batlow colour scale
cm_data = np.loadtxt("batlow.txt")


def plot_spct_series(
    S: np.ndarray, f: np.ndarray, t: np.ndarray, norm: str = None,
    norm_method: str = None, title: str = None, outfile=None, fmt='pdf',
    dpi=300, flim: Tuple[int, int] = None,
        tlim: Tuple[datetime, datetime] = None):
    """
    Plots a spectral series.

    :param S: A spectral series with dim=2.
    :type S: np.ndarray
    :param f: Vector containing frequencies (Hz)
    :type f: np.ndarray
    :param t: Vector containing times (in s)
    :type t: np.ndarray
    :param norm: Normalise the spectrum either on the time axis with
        norm='t' or on the frequency axis with norm='f', defaults to None.
    :type norm: str, optional
    :param norm_method: Normation method to use.
        Either 'linalg' (i.e., length of vector),
        'mean', or 'median'.
    :param title: Plot title, defaults to None
    :type title: str, optional
    :param outfile: location to save the plot to, defaults to None
    :type outfile: str, optional
    :param fmt: Format to save figure, defaults to 'pdf'
    :type fmt: str, optional
    :param flim: Limit Frequency axis and Normalisation to the values
        in the given window
    :type flim: Tuple[int, int]
    :param tlim: Limit time axis to the values in the given window
    :type tlim: Tuple[datetime, datetime]
    """
    # Mask nans
    S = np.ma.masked_invalid(S)

    # Show dates in English format
    locale.setlocale(locale.LC_ALL, "en_GB.utf8")
    # Create UTC time series
    utc = []
    for pit in t:
        utc.append(UTCDateTime(pit).datetime)
    del t

    set_mpl_params()

    if log_scale:
        plt.yscale('log')

    if flim is not None:
        plt.ylim(flim)
        ii = np.argmin(abs(f-flim[0]))
        jj = np.argmin(abs(f-flim[1])) + 1
        f = f[ii:jj]
        S = S[ii:jj, :]
    else:
        plt.ylim(10**-1, f.max())

    if tlim is not None:
        plt.xlim(tlim)
        utc = np.array(utc)
        ii = np.argmin(abs(utc-tlim[0]))
        jj = np.argmin(abs(utc-tlim[1]))
        utc = utc[ii:jj]
        S = S[:, ii:jj]

    # Normalise
    if not norm:
        pass
    elif norm == 'f':
        if norm_method == 'linalg':
            S = np.divide(S, np.linalg.norm(S, axis=1)[:, np.newaxis])
        elif norm_method == 'mean':
            S = np.divide(S, np.mean(S, axis=1)[:, np.newaxis])
        elif norm_method == 'median':
            S = np.divide(S, np.median(S, axis=1)[:, np.newaxis])
        else:
            raise ValueError('Normalisation method %s unkown.' % norm_method)
    elif norm == 't':
        if norm_method == 'linalg':
            S = np.divide(S, np.linalg.norm(S, axis=0))
        elif norm_method == 'mean':
            S = np.divide(S, np.mean(S, axis=0))
        elif norm_method == 'median':
            S = np.divide(S, np.median(S, axis=0))
        else:
            raise ValueError('Normalisation method %s unkown.' % norm_method)
    else:
        raise ValueError('Normalisation %s unkown.' % norm)

    cmap = colors.LinearSegmentedColormap.from_list('batlow', cm_data)
    cmap.set_bad('k')
    S /= S.max()
    pcm = plt.pcolormesh(
        utc, f, S, shading='gouraud', cmap=cmap,
        norm=colors.LogNorm()#vmin=1e-10, vmax=1e-5)
        )
    plt.colorbar(
        pcm, label='energy (normalised)', orientation='horizontal', shrink=.6)
    plt.ylabel(r'$f$ [Hz]')
    # plt.xlabel('(dd/mm)')
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%h %y'))
    plt.xticks(rotation=35)
    plt.title(f'{network}.{station}.{channel[-1]}')
    return ax


def spct_series_welch(streams: Generator[Stream], window_length: int):
    """
    Computes a spectral time series. Each point in time is computed using the
    welch method. Windows overlap by half the windolength. The input stream can
    contain one or several traces from the same station. Frequency axis is
    logarithmic.
    :param st: Input Stream with data from one station.
    :type st: ~obspy.core.Stream
    :param window_length: window length in seconds for each datapoint in time
    :type window_length: int or float
    :return: Arrays containing a frequency and time series and the spectral
        series.
    :rtype: np.ndarray
    """
    specl = []
    # List of actually available times
    t = []
    for st in streams:
        st = st.merge()
        tr = preprocess(st[0])
        for wintr in tr.slide(window_length=window_length, step=window_length):
            f, S = welch(wintr.data, fs=tr.stats.sampling_rate)
            # interpolate onto a logarithmic frequency space
            # 512 points of resolution in f direction hardcoded for now
            f2 = np.logspace(-3, np.log10(f.max()), 512)
            S2 = pchip_interpolate(f, S, f2)
            specl.append(S2)
            t.append(tr.stats.starttime)
    S = np.array(specl)

    t = np.array(t)
    return f2, t, S.T


def preprocess(tr: Trace):
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
    resample_or_decimate(tr, 25)
    tr.remove_response()
    # Detrend
    tr.detrend(type='linear')

    # highpass filter
    tr.filter('bandpass', freqmin=0.01, freqmax=12)

    return tr


def set_mpl_params():
    params = {
        #'font.family': 'Avenir Next',
        'pdf.fonttype': 42,
        'font.weight': 'bold',
        'figure.dpi': 150,
        'axes.labelweight': 'bold',
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 13,
        'xtick.direction': 'in',
        'xtick.top': True,  # draw label on the top
        'xtick.bottom': True,  # draw label on the bottom
        'xtick.minor.visible': True,
        'xtick.major.top': True,  # draw x axis top major ticks
        'xtick.major.bottom': True,  # draw x axis bottom major ticks
        'xtick.minor.top': True,  # draw x axis top minor ticks
        'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'ytick.labelsize': 13,
        'ytick.direction': 'in',
        'ytick.left': True,  # draw label on the top
        'ytick.right': True,  # draw label on the bottom
        'ytick.minor.visible': True,
        'ytick.major.left': True,  # draw x axis top major ticks
        'ytick.major.right': True,  # draw x axis bottom major ticks
        'ytick.minor.left': True,  # draw x axis top minor ticks
        'ytick.minor.right': True,  # draw x axis bottom minor ticks
        'legend.fancybox': False,
        'legend.frameon': False,
        'legend.loc': 'upper left',
        'legend.numpoints': 2,
        'legend.fontsize': 'large',
        'legend.framealpha': 1,
        'legend.scatterpoints': 3,
        'legend.edgecolor': 'inherit'
    }
    matplotlib.rcParams.update(params)
    # matplotlib.font_manager._rebuild()
