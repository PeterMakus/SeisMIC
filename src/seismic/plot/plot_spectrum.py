'''
Plotting specral time series.

:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 21st June 2023 04:54:20 pm
Last Modified: Tuesday, 1st April 2025 10:12:48 am
'''

import os
from datetime import datetime
from typing import Tuple
import locale

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import colors
from obspy import UTCDateTime

from seismic.plot.plot_utils import set_mpl_params

# Batlow colour scale
batlow = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'batlow.txt')


def plot_spct_series(
    S: np.ndarray, f: np.ndarray, t: np.ndarray, norm: str = None,
    norm_method: str = None, flim: Tuple[float, float] | float = None,
    tlim: Tuple[datetime, datetime] = None, cmap: str = 'batlow', vmin=None,
        vmax=None, log_scale: bool = False) -> plt.Axes:
    """
    Plots a spectral series.

    :param S: A spectral series with dim=2.
    :type S: np.ndarray
    :param f: Vector containing frequencies (Hz)
    :type f: np.ndarray
    :param t: Vector containing times as UTCDateTimes
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
        in the given window. Can be a tuple of floats or a single float.
        If a single float is given, a 2D plot is created.
    :type flim: Tuple[int, int] | float, optional
    :param tlim: Limit time axis to the values in the given window
    :type tlim: Tuple[datetime, datetime], optional
    :param cmap: Colormap to use, defaults to 'batlow'
    :type cmap: str, optional
    :param vmin: Minimum value for color normalization, defaults to None
    :type vmin: float, optional
    :param vmax: Maximum value for color normalization, defaults to None
    :type vmax: float, optional
    :param log_scale: Use logarithmic scale for the color map,
        defaults to False
    :type log_scale: bool, optional
    :return: The axis object of the plot.
    :rtype: plt.Axes
    """
    # Mask nans
    S = np.ma.masked_invalid(S)

    # Show dates in English format
    try:
        locale.setlocale(locale.LC_ALL, "en_GB.utf8")
    # for MAC-OS the name is slightly different -_-
    except locale.Error:
        locale.setlocale(locale.LC_ALL, "en_GB.UTF-8")
    # Create UTC time series

    set_mpl_params()

    if log_scale:
        plt.yscale('log')
    if isinstance(flim, (float, int)):
        # Call 1D plotting function
        return _plot_spct_series_1d(S, f, t, flim, tlim)
    elif flim is not None:
        plt.ylim(flim)
        ii = np.argmin(abs(f-flim[0]))
        jj = np.argmin(abs(f-flim[1])) + 1
        f = f[ii:jj]
        S = S[ii:jj, :]
    else:
        plt.ylim(.1, f.max())

    t = np.array([UTCDateTime(tt).datetime for tt in t])

    if tlim is not None:
        plt.xlim(tlim)
        ii = np.argmin(abs(t-tlim[0]))
        jj = np.argmin(abs(t-tlim[1]))
        t = t[ii:jj]
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
    if cmap == 'batlow':
        cmap = colors.LinearSegmentedColormap.from_list(
            'batlow', np.loadtxt(batlow))
        cmap.set_bad('k')
    S /= S.max()
    pcm = plt.pcolormesh(
        t, f, S, shading='gouraud', cmap=cmap,
        norm=colors.LogNorm(), vmin=vmin, vmax=vmax)
    plt.colorbar(
        pcm, label='Energy (normalised)', orientation='horizontal', shrink=.6,
        pad=.25)
    plt.ylabel(r'$f$ [Hz]')
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(
        mdates.AutoDateFormatter(ax.get_axes_locator()))
    plt.xticks(rotation=35)
    return ax


def _plot_spct_series_1d(
    S: np.ndarray, f: np.ndarray, t: np.ndarray,
        freq: float, tlim: Tuple[datetime, datetime] = None) -> plt.Axes:
    """
    Plots a 1D spectral series for a specific frequency.

    :param S: A spectral series with dim=2.
    :type S: np.ndarray
    :param f: Vector containing frequencies (Hz)
    :type f: np.ndarray
    :param t: Vector containing times as UTCDateTimes
    :type t: np.ndarray
    :param freq: Specific frequency to plot
    :type freq: float
    :param tlim: Limit time axis to the values in the given window
    :type tlim: Tuple[datetime, datetime], optional
    """
    # Find the index of the closest frequency
    freq_idx = np.argmin(abs(f - freq))

    # Extract the spectral series for the specific frequency
    S_1d = S[freq_idx, :]

    # Convert times to datetime
    t = np.array([UTCDateTime(tt).datetime for tt in t])

    # Apply time limits if provided
    if tlim is not None:
        plt.xlim(tlim)
        ii = np.argmin(abs(t - tlim[0]))
        jj = np.argmin(abs(t - tlim[1]))
        t = t[ii:jj]
        S_1d = S_1d[ii:jj]

    set_mpl_params()

    # Plot the 1D spectral series
    plt.plot(t, S_1d, label=f'{freq:.2f} Hz')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(
        mdates.AutoDateFormatter(ax.get_axes_locator()))
    plt.xticks(rotation=35)
    return ax
