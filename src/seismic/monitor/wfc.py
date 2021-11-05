'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 5th November 2021 08:19:58 am
Last Modified: Friday, 5th November 2021 08:52:59 am
'''
import numpy as np


class WFC(dict):
    def __init__(self, wfc_dict: dict):
        for k, v in wfc_dict.items():
            self[k] = v
        del wfc_dict
        # dict to hold averages
        self.av = {}

    def compute_average(self):
        for k, v in self.items():
            self.av['%s_av' % k] = np.mean(v, axis=0)

    def save(self, path: str):
        np.savez_compressed(path, **self.av, **self)


def read_wfc(path: str) -> WFC:
    loaded = np.load(path)
    d = {}
    av = {}
    for k, v in loaded.items():
        if k[-3:] == '_av':
            av[k] = v
        else:
            d[k] = v
    wfc = WFC(d)
    wfc.av = av
    return wfc
