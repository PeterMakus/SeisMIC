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
Last Modified: Tuesday, 11th July 2023 01:46:20 pm
'''
import os

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import colors

from seismic.plot.plot_utils import set_mpl_params


# Batlow colour scale
batlow = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'batlow.txt')


def plot_wfc_bulk(
    time_lag: np.ndarray, cfreq: np.ndarray, wfc: np.ndarray, title: str,
        log: bool, cmap: str):
    set_mpl_params()
    if cmap == 'batlow':
        cmap = colors.LinearSegmentedColormap.from_list(
            'batlow', np.loadtxt(batlow))
        cmap.set_bad('k')
    cmesh = plt.pcolormesh(time_lag, cfreq, wfc, cmap=cmap)
    plt.xlabel(r"Central Lag Time [s]")
    plt.ylabel(r"Centre Frequency [Hz]")
    plt.title(title)
    plt.colorbar(cmesh, label='Coherence')
    if log:
        plt.yscale('log')
    return plt.gca()
