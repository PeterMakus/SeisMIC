'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 15th June 2021 04:12:18 pm
Last Modified: Tuesday, 15th June 2021 04:12:31 pm
'''

import numpy as np


class DV(object):
    def __init__(
        self, corr: np.ndarray, value: np.ndarray, sim_mat: np.ndarray,
            second_axis: np.ndarray, method: str):
        # Allocate attributes
        self.corr = corr
        self.value = value
        self.sim_mat = sim_mat
        self.second_axis = second_axis
        self.method = method
