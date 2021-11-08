'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 5th November 2021 08:19:58 am
Last Modified: Monday, 8th November 2021 09:42:49 am
'''
import numpy as np

from seismic.correlate.stats import CorrStats
from seismic.utils import miic_utils as mu


class WFC(dict):
    def __init__(self, wfc_dict: dict, stats: CorrStats):
        for k, v in wfc_dict.items():
            self[k] = v
        del wfc_dict
        # dict to hold averages
        self.av = {}
        self.stats = stats

    def compute_average(self):
        for k, v in self.items():
            self.av['%s_av' % k] = np.nanmean(v, axis=0)

    def save(self, path: str):
        kwargs = mu.save_header_to_np_array(self.stats)
        np.savez_compressed(path, **self.av, **self, **kwargs)


def read_wfc(path: str) -> WFC:
    loaded = np.load(path)
    d = {}
    av = {}
    for k, v in loaded.items():
        if k[-3:] == '_av':
            av[k] = v
        else:
            d[k] = v
    wfc = WFC(d, stats=mu.load_header_from_np_array(loaded))
    wfc.av = av
    return wfc
