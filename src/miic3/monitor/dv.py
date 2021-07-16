'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 15th June 2021 04:12:18 pm
Last Modified: Friday, 16th July 2021 02:36:59 pm
'''

from datetime import datetime
from typing import List, Tuple
import numpy as np
from miic3.plot.plot_dv import plot_dv

from miic3.utils.miic_utils import save_header_to_np_array, \
    load_header_from_np_array
from miic3.correlate.stats import CorrStats


class DV(object):
    """
    A simple object to save velocity changes.
    """
    def __init__(
        self, corr: np.ndarray, value: np.ndarray, value_type: str,
        sim_mat: np.ndarray, second_axis: np.ndarray, method: str,
            stats: CorrStats):
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

    def plot(
        self, save_dir: str = '.', figure_file_name: str = None,
        mark_time: datetime = None, normalize_simmat: bool = False,
        sim_mat_Clim: List[float] = [],
            figsize: Tuple[float, float] = (9, 11), dpi: int = 72):
        plot_dv(
            self.__dict__, save_dir, figure_file_name, mark_time,
            normalize_simmat, sim_mat_Clim, figsize, dpi)


def read_dv(path: str) -> DV:
    """
    Reads a saved velocity change object from an **.npz** file.

    :param path: Path to file
    :type path: str
    :return: the corresponding and converted DV object
    :rtype: DV
    """
    loaded = np.load(path)
    stats = CorrStats(load_header_from_np_array(loaded))
    return DV(
        loaded['corr'], loaded['value'], loaded['vt_array'][0],
        loaded['sim_mat'], loaded['second_axis'], loaded['method_array'][0],
        stats=stats)
