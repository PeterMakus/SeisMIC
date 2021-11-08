'''
:copyright:
    Module to plot waveform coherencies.
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 8th November 2021 02:46:15 pm
Last Modified: Monday, 8th November 2021 02:58:43 pm
'''

from matplotlib import pyplot as plt
import numpy as np

from seismic.plot.plot_utils import set_mpl_params


def plot_wfc_bulk(
        time_lag: np.ndarray, cfreq: np.ndarray, wfc: np.ndarray, title: str):
    set_mpl_params()
    plt.pcolormesh(time_lag, cfreq, wfc)
    plt.xlabel(r"Lag Time [s]")
    plt.ylabel(r"Centre Frequency [Hz]")
    plt.title(title)
