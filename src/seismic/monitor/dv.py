'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 15th June 2021 04:12:18 pm
Last Modified: Monday, 21st February 2022 02:17:52 pm
'''

from datetime import datetime
from glob import glob
from typing import List, Tuple
import warnings
from zipfile import BadZipFile

import numpy as np

from seismic.plot.plot_dv import plot_dv
from seismic.utils import miic_utils as mu
from seismic.correlate.stats import CorrStats


class DV(object):
    """
    A simple object to save velocity changes.
    """
    def __init__(
        self, corr: np.ndarray, value: np.ndarray, value_type: str,
        sim_mat: np.ndarray, second_axis: np.ndarray, method: str,
        stats: CorrStats, std_val: np.ndarray = None,
            std_corr: np.ndarray = None):
        """
        Creates an object designed to hold and process velocity changes.

        :param corr: Array with maximum Correlation coefficients.
        :type corr: np.ndarray
        :param value: Array with decimal velocity changes / stretches.
        :type value: np.ndarray
        :param value_type: Type of value (e.g., stretch)
        :type value_type: str
        :param sim_mat: Similarity Matrix with the dimensions time and stretch.
        :type sim_mat: np.ndarray
        :param second_axis: second axis (e.g., the stretch axis)
        :type second_axis: np.ndarray
        :param method: Stretching method that was used
        :type method: str
        :param stats: Stats of the correlation object that was used
        :type stats: CorrStats
        :param std_val: Standard deviation of the value (e.g., stretch).
            In case this dv object holds an average
            of several dvs, the user can choose to compute the standard
            deviation over all dvs' values. Same shape as corr and
            value, defaults to None
        :type std_val: np.ndarray, optional
        :param std_corr: Standard deviation of the correlation coefficient.
            In case this dv object holds an average
            of several dvs, the user can choose to compute the standard
            deviation over all dvs' values. Same shape as corr and
            value, defaults to None
        :type std_corr: np.ndarray, optional
        """
        # Allocate attributes
        self.value_type = value_type
        self.corr = corr
        self.value = value
        self.sim_mat = sim_mat
        self.std_val = std_val
        self.std_corr = std_corr
        self.second_axis = second_axis
        self.method = method
        self.stats = stats

    def __str__(self):
        """
        Print a prettier string.
        """
        code = f'{self.stats.network}.{self.stats.station}.'\
            + self.stats.channel
        out = f'{self.method} {self.value_type} velocity change estimate of '\
            + f'{code}.\nstarttdate: {min(self.stats.corr_start).ctime()}\n'\
            + f'enddate: {max(self.stats.corr_end).ctime()}'
        return out

    def save(self, path: str):
        """
        Saves the dv object to a compressed numpy binary format. The DV can
        later be read using :func:`~seismic.monitor.dv.read_dv`.

        :param path: output path
        :type path: str
        """
        method_array = np.array([self.method])
        vt_array = np.array([self.value_type])
        kwargs = mu.save_header_to_np_array(self.stats)
        np.savez_compressed(
            path, corr=self.corr, value=self.value, sim_mat=self.sim_mat,
            second_axis=self.second_axis, method_array=method_array,
            vt_array=vt_array, std_val=self.std_val, std_corr=self.std_corr,
            **kwargs)

    def plot(
        self, save_dir: str = '.', figure_file_name: str = None,
        mark_time: datetime = None, normalize_simmat: bool = False,
        sim_mat_Clim: List[float] = [], ylim: Tuple[int, int] = None,
        plot_std: bool = False, figsize: Tuple[float, float] = (9, 11),
            dpi: int = 144, title: str = None):
        """
        Plots the dv object into a *multi-panel-view* of `similarity matrix`
        `dv-value`, and `correlation coefficient`.

        :param save_dir: Directory to save the figure to, defaults to '.'
        :type save_dir: str, optional
        :param figure_file_name: Filename of the figure. If None, the figure
            will not be save, defaults to None.
        :type figure_file_name: str, optional
        :param mark_time: times to put ticks to. If not set, they will be
            determined automatically, defaults to None.
        :type mark_time: datetime, optional
        :param normalize_simmat: Normalise the similarity matrix?
            Defaults to False.
        :type normalize_simmat: bool, optional
        :param sim_mat_Clim: Color limits for the similarity matrix
            , defaults to []. Format is [low_limit, upper_limit]
        :type sim_mat_Clim: List[float], optional
        :param ylim: Y-limits (e.g., stretch) of the plot, defaults to None
        :type ylim: Tuple[int, int], optional
        :param plot_std: If set to True, the upper and lower limit of the
            value's standard deviation will be plotted as bounds,
            defaults to False.
        :type plot_std: bool, optional
        :param figsize: Size of the figure/canvas, defaults to (9, 11)
        :type figsize: Tuple[float, float], optional
        :param dpi: Pixels per inch, defaults to 144
        :type dpi: int, optional
        :param title: Define a custom title. Otherwise, an automatic title
            will be created, defaults to None
        :type title: str, optional
        """
        plot_dv(
            self.__dict__, save_dir, figure_file_name, mark_time,
            normalize_simmat, sim_mat_Clim, figsize, dpi, ylim=ylim,
            title=title, plot_std=plot_std)

    def smooth_sim_mat(self, win_len: int):
        """
        Smoothes the similarity matrix along the correlation time axis.

        :param win_len: Length of the window in number of samples.
        :type win_len: int

        :note::

            This action is perfomed in-place.
        """
        self.sim_mat = mu.nan_moving_av(self.sim_mat, int(win_len/2), axis=0)

        # Compute the dependencies again
        self.corr = np.nanmax(self.sim_mat, axis=1)
        self.value = self.second_axis[
            np.argmax(np.nan_to_num(self.sim_mat), axis=1)]
        return self


def read_dv(path: str) -> DV:
    """
    Reads a saved velocity change object from an **.npz** file.

    :param path: Path to file, wildcards allowed
    :type path: str
    :return: the corresponding and converted DV object
    :rtype: DV, List[DV] if path contains wildcards
    """
    if '*' in path or '?' in path:
        dvl = []
        for p in glob(path):
            try:
                dvl.append(read_dv(p))
            except(BadZipFile):
                warnings.warn(f'File {p} corrupt, skipping..')
        if not len(dvl):
            raise FileNotFoundError(
                f'No files that adhere the pattern {path} found.')
        return dvl
    loaded = np.load(path)
    stats = CorrStats(mu.load_header_from_np_array(loaded))
    # to check that compatibility works
    vt = loaded['vt_array']
    while not isinstance(vt, str):
        vt = vt[0]
    method = loaded['method_array']
    while not isinstance(method, str):
        method = method[0]
    try:
        std_val = loaded['std_val']
        std_corr = loaded['std_corr']
    except KeyError:
        std_val = std_corr = None
    return DV(
        loaded['corr'], loaded['value'], vt, loaded['sim_mat'],
        loaded['second_axis'], method, stats=stats, std_val=std_val,
        std_corr=std_corr)
