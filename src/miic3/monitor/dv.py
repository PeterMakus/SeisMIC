'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 15th June 2021 04:12:18 pm
Last Modified: Wednesday, 16th June 2021 05:07:07 pm
'''

import numpy as np


class DV(object):
    """
    A simple object to save velocity changes.
    """
    def __init__(
        self, corr: np.ndarray, value: np.ndarray, value_type: str,
            sim_mat: np.ndarray, second_axis: np.ndarray, method: str):
        # Allocate attributes
        self.value_type = value_type
        self.corr = corr
        self.value = value
        self.sim_mat = sim_mat
        self.second_axis = second_axis
        self.method = method

    def save(self, path: str):
        method_array = np.array([self.method])
        vt_array = np.array([self.value_type])
        np.savez_compressed(
            path, corr=self.corr, value=self.value, sim_mat=self.sim_mat,
            second_axis=self.second_axis, method_array=method_array,
            vt_array=vt_array)


def read_dv(path: str) -> DV:
    loaded = np.load(path)
    return DV(
        loaded['corr'], loaded['value'], loaded['vt_array'][0],
        loaded['sim_mat'], loaded['second_axis'], loaded['method_array'][0])
