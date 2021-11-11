'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 5th November 2021 08:19:58 am
Last Modified: Wednesday, 10th November 2021 11:52:50 am
'''
import glob
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
        nec_keys = ['tw_start', 'tw_len', 'freq_min', 'freq_max']
        if not all(k in stats for k in nec_keys):
            raise ValueError(f'Stats has to contain the keys {nec_keys}.')
        for k, v in wfc_dict.items():
            self[k] = v
        del wfc_dict
        # dict to hold averages
        self.av = {}
        self.stats = stats
        self.mean = np.array([])

    def compute_average(self):
        """
        Computes the average of the waveform coherency over the time axis.

        :param average_reftrcs: Average over all reference traces. Defaults to
            True.
        :type average_reftrcs: bool, optional
        """
        for k, v in self.items():
            self.av['%s_av' % k] = np.nanmean(v)
        self.mean = np.nanmean([v for v in self.av.values()])

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
            lw.append(wfc.stats.tw_start + wfc.stats.tw_len/2)
            means.append(wfc.mean)
        # create grids
        lwg, freqg = np.meshgrid(sorted(set(lw)), sorted(set(freq)))
        # actual wfc grid
        self.wfc = np.full_like(lwg, np.nan)
        for f, lwc, v in zip(freq, lw, means):
            ii = np.argwhere(((lwg == lwc) & (freqg == f)))[0]
            self.wfc[ii[0], ii[1]] = v
        del means
        # create grids
        self.lw = np.array(sorted(set(lw)))
        self.lwg = lwg
        del lw
        self.cfreq = np.array(sorted(set(freq)))
        self.cfreqg = freqg
        del freq

    def plot(self, title: str = None, outfile: str = None):
        plot_wfc_bulk(
            self.lw, self.cfreq, self.wfc, title=title, outfile=outfile)


def read_wfc(path: str) -> WFC:
    """
    Read a :class:`~seismic.monitor.wfc.WFC` object saved as *npz*.
    Returns a list if wildcards are used

    :param path: Path to file
    :type path: str
    :return: The WFC object.
    :rtype: WFC
    """
    if '*' in path:
        return [read_wfc(f) for f in glob.glob(path, recursive=True)]
    loaded = np.load(path)
    d = {}
    av = {}
    stats = {}
    for k, v in loaded.items():
        if k[-3:] == '_av':
            av[k] = v
        elif k == 'mean':
            mean = v
        elif 'reftr' in k:
            d[k] = v
        else:
            stats[k] = v
    wfc = WFC(d, stats=CorrStats(mu.load_header_from_np_array(stats)))
    wfc.av = av
    wfc.mean = mean
    return wfc
