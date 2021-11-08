'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 5th November 2021 08:19:58 am
Last Modified: Monday, 8th November 2021 03:03:06 pm
'''
from typing import List
import numpy as np

from seismic.correlate.stats import CorrStats
from seismic.utils import miic_utils as mu
from seismic.plot.plot_wfc import plot_wfc_bulk


class WFC(dict):
    """
    Object to hold the properties of a waveform coherency for a single
    frequency and lapse time window.

    The waveform coherency can be used as a measure of stability of
    a certain correlation. See Steinmann, et. al. (2021) for details.
    """
    def __init__(self, wfc_dict: dict, stats: CorrStats):
        """
        Initialise WFC object.

        :param wfc_dict: Dictionary holding the correlation values for
            each of the reference traces. The keys follow the syntax
            `reftr_%n` where n is the number of the reference trace
            (starting with 0).
        :type wfc_dict: dict
        :param stats: CorrStats object from the original CorrBulk.
            Also has to contain the keys holding the information about the
            bandpass filter and the used lapse time window.
        :type stats: CorrStats
        """
        nec_keys = ['tw_start', 'tw_length', 'freq_min', 'freq_max']
        if not all(k in wfc_dict for k in nec_keys):
            raise ValueError(f'Stats has to contain the keys {nec_keys}.')
        for k, v in wfc_dict.items():
            self[k] = v
        del wfc_dict
        # dict to hold averages
        self.av = {}
        self.stats = stats
        self.mean = np.array([])

    def compute_average(self, average_reftrcs: bool = True):
        """
        Computes the average of the waveform coherency over the time axis.

        :param average_reftrcs: Average over all reference traces. Defaults to
            True.
        :type average_reftrcs: bool, optional
        """
        for k, v in self.items():
            self.av['%s_av' % k] = np.nanmean(v, axis=0)
        self.mean = np.nanmean([v for v in self.av.values()], axis=0)

    def save(self, path: str):
        """
        Saves the object to an npz file.

        :param path: Path to save to.
        :type path: str
        """
        kwargs = mu.save_header_to_np_array(self.stats)
        np.savez_compressed(path, mean=self.mean, **self.av, **self, **kwargs)


class WFCBulk(object):
    """
    Object to hold the waveform coherency for one station and subsequently plot
    it against frequency and lapse time window.
    """
    def __init__(self, wfcl: List[WFC]):
        # Compute averages if not available
        freq = []
        lw = []
        means = []
        for wfc in wfcl:
            wfc.mean.size or wfc.compute_average()
            # Midpoint of bp-filter (centre frequency)
            freq.append((wfc.stats.freq_max + wfc.stats.freq_min) / 2)
            # midpoint of lapse window
            lw.append(wfc.stats.tw_start + wfc.stats.tw_length/2)
            means.append(wfc.mean)
        freq, lw, means = zip(*sorted(zip(lw, freq, means)))
        # create grids
        self.lw = np.array(list(set(lw)))
        del lw
        self.cfreq = np.array(list(set(freq)))
        del freq
        # Now the lapse window changes per column and freq per row
        self.wfc = np.reshape(means, (len(self.cfreq), len(self.lw)))
        del means

    def plot(self, title: str = None):
        plot_wfc_bulk(self.lw, self.cfreq, self.wfc, title=title)


def read_wfc(path: str) -> WFC:
    """
    Read a :class:`~seismic.monitor.wfc.WFC` object saved as *npz*.

    :param path: Path to file
    :type path: str
    :return: The WFC object.
    :rtype: WFC
    """
    loaded = np.load(path)
    d = {}
    av = {}
    for k, v in loaded.items():
        if k[-3:] == '_av':
            av[k] = v
        elif k == 'mean':
            mean = v
        else:
            d[k] = v
    wfc = WFC(d, stats=mu.load_header_from_np_array(loaded))
    wfc.av = av
    wfc.mean = mean
    return wfc
