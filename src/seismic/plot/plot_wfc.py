'''
:copyright:
    Module to plot waveform coherencies.
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 8th November 2021 02:46:15 pm
Last Modified: Monday, 16th January 2023 11:13:58 am
'''

from matplotlib import pyplot as plt
import numpy as np

from seismic.plot.plot_utils import set_mpl_params


def plot_wfc_bulk(
    time_lag: np.ndarray, cfreq: np.ndarray, wfc: np.ndarray, title: str,
        outfile: str):
    set_mpl_params()
    cmesh = plt.pcolormesh(time_lag, cfreq, wfc)
    plt.xlabel(r"Lag Time [s]")
    plt.ylabel(r"Centre Frequency [Hz]")
    plt.title(title)
    plt.colorbar(cmesh, label='Correlation Coefficient')
    if outfile:
        plt.tight_layout()
        plt.savefig(outfile, format='png', dpi=300)
