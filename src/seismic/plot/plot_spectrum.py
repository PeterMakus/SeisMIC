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
Last Modified: Monday, 21st August 2023 12:22:56 pm
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
    norm_method: str = None, flim: Tuple[int, int] = None,
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
