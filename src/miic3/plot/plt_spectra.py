'''
Simple script to compute and plot time-dependent spectral power densities.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 15th February 2021 02:09:48 pm
Last Modified: Friday, 30th April 2021 09:39:43 am
'''

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import numpy as np
from obspy import UTCDateTime
import obspy
from scipy.signal import welch
from scipy.interpolate import pchip_interpolate


def plot_spct_series(
    S: np.ndarray, f: np.ndarray, t: np.ndarray, norm: str = None,
    norm_method: str = None, title: str = None, outfile=None, fmt='pdf',
        dpi=300):
    """
    Plots a spectral series.

    :param S: A spectral series with dim=2.
    :type S: np.ndarray
    :param f: Vector containing frequencies (Hz)
    :type f: np.ndarray
    :param t: Vector containing times (in s)
    :type t: np.ndarray
    :param norm: Normalise the spectrum either on the time axis with
        `norm='t'` or on the frequency axis with `norm='f'`, defaults to None.
    :type norm: str, optional
    :param norm_method: Normation method to use. Either `'linalg'`
        (i.e., length of vector), `'mean'`, or `'median'`.
    :param title: Plot title, defaults to None
    :type title: str, optional
    :param outfile: location to save the plot to, defaults to None
    :type outfile: str, optional
    :param fmt: Format to save figure, defaults to 'pdf'
    :type fmt: str, optional
    """
    # Create UTC time series
    utc = []
    for pit in t:
        utc.append(UTCDateTime(pit).datetime)
    del t

    set_mpl_params()

    plt.yscale('log')
    plt.ylim(10**-2, f.max())

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

    pcm = plt.pcolormesh(
        utc, f, S, shading='gouraud', norm=colors.LogNorm(
            vmin=S.min(), vmax=S.max()))
    plt.colorbar(pcm)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('(dd/mm)')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    if title:
        plt.title(title)
    if outfile:
        if fmt == 'pdf' or fmt == 'svg':
            plt.savefig(outfile+'.'+fmt, format=fmt)
        else:
            plt.savefig(outfile+'.'+fmt, format=fmt, dpi=dpi)
    else:
        plt.show()


def spct_series_welch(
        st: obspy.Stream, window_length: float) -> np.ndarray:
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
    l = []
    st.sort(keys=['starttime'])
    for tr in st:
        for wintr in tr.slide(
            window_length=window_length, step=window_length+tr.stats.delta,
                include_partial_windows=True):
            # windows will overlap with half the window length
            # Hard-corded nperseg so that the longest period waves that
            # can be resolved are around 300s
            f, S = welch(wintr.data, fs=tr.stats.sampling_rate, nperseg=2**15)

            # interpolate onto a logarithmic frequency space
            # 256 points of resolution in f direction hardcoded for now
            f2 = np.logspace(-3, np.log10(f.max()), 256)
            S2 = pchip_interpolate(f, S, f2)
            l.append(S2)
    S = np.array(l)

    # compute time series
    t = np.linspace(
        st[0].stats.starttime.timestamp, st[-1].stats.endtime.timestamp,
        S.shape[0])
    return f2, t, S.T


def set_mpl_params():
    params = {
        # 'font.family': 'Avenir Next',
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
    matplotlib.font_manager._rebuild()
