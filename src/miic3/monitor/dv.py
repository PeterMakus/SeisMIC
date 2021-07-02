'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 15th June 2021 04:12:18 pm
Last Modified: Friday, 18th June 2021 03:40:18 pm
'''

from typing import List

import numpy as np
from obspy.core.trace import Stats
from obspy.core.utcdatetime import UTCDateTime

from miic3.utils.miic_utils import save_header_to_np_array, \
    load_header_from_np_array


class DV(object):
    """
    A simple object to save velocity changes.
    """
    def __init__(
        self, corr: np.ndarray, value: np.ndarray, value_type: str,
        sim_mat: np.ndarray, second_axis: np.ndarray, method: str,
            stats: Stats):
        # Allocate attributes
        self.value_type = value_type
        self.corr = corr
        self.value = value
        self.sim_mat = sim_mat
        self.second_axis = second_axis
        self.method = method
        # self.corr_start = stats.corr_start
        self.stats = stats

    def save(self, path: str):
        """
        Saves the dv object to a compressed numpy binary format. The DV can
        later be read using :func:`~miic3.monitor.dv.read_dv`.

        :param path: output path
        :type path: str
        """
        method_array = np.array([self.method])
        vt_array = np.array([self.value_type])
        kwargs = save_header_to_np_array(self.stats)
        np.savez_compressed(
            path, corr=self.corr, value=self.value, sim_mat=self.sim_mat,
            second_axis=self.second_axis, method_array=method_array,
            vt_array=vt_array, **kwargs)


def read_dv(path: str) -> DV:
    """
    Reads a saved velocity change object from an **.npz** file.

    :param path: Path to file
    :type path: str
    :return: the corresponding and converted DV object
    :rtype: DV
    """
    loaded = np.load(path)
    stats = load_header_from_np_array(loaded)
    return DV(
        loaded['corr'], loaded['value'], loaded['vt_array'][0],
        loaded['sim_mat'], loaded['second_axis'], loaded['method_array'][0],
        stats=stats)