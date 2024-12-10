'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 15th June 2021 04:12:18 pm
Last Modified: Monday, 9th December 2024 02:15:00 pm
'''

from copy import deepcopy
from datetime import datetime, time
from glob import glob
from typing import List, Tuple, Optional
import warnings
from zipfile import BadZipFile

import matplotlib.pyplot as plt
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
        stats: CorrStats, stretches: np.ndarray = None,
        corrs: np.ndarray = None, n_stat: np.ndarray = None,
            dv_processing: dict = {}):
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
        :param stretches: Scatter of the value (e.g., stretch).
            In case this dv object holds an average
            of several dvs, the user can choose to keep all dvs' stretches.
            defaults to None
        :type stretches: np.ndarray, optional
        :param corrs: Scatter of the correlation coefficient (e.g., stretch).
            In case this dv object holds an average
            of several dvs, the user can choose to keep all dvs' corrs.
            defaults to None
        :type corrs: np.ndarray, optional
        :param n_stat: Number of stations used for the stack at corr_start t.
            Has the same shape as `value` and `corr`. Defaults to None
        :type n_stat: np.ndarray, optional
        """
        # Allocate attributes
        self.value_type = value_type
        self.corr = corr
        self.value = value
        self.sim_mat = sim_mat
        self.corrs = corrs
        self.stretches = stretches
        self.n_stat = n_stat
        self.second_axis = second_axis
        self.method = method
        self.stats = stats
        self.dv_processing = dv_processing
        # Available indices
        self.avail = ~np.isnan(self.corr)

    def __str__(self):
        """
        Print a prettier string.
        """
        code = f'{self.stats.network}.{self.stats.station}.'\
            + f'{self.stats.location}.{self.stats.channel}'
        out = f'{self.method} {self.value_type} velocity change estimate of '\
            + f'{code}.\nstarttdate: {min(self.stats.corr_start).ctime()}\n'\
            + f'enddate: {max(self.stats.corr_end).ctime()}\n\n'\
            + f'processed with the following parameters: {self.dv_processing}'
        if self.method == np.array(['time_shift']):
            out = f'Time shift estimate of {code}.\nstarttdate: '\
                + f'{min(self.stats.corr_start).ctime()}\nenddate: '\
                + f'{max(self.stats.corr_end).ctime()}'
        return out

    def save(self, path: str):
        """
        Saves the dv object to a compressed numpy binary format. The DV can
        later be read using :func:`~seismic.monitor.dv.read_dv`.

        :param path: output path
        :type path: str
        """
        kwargs = mu.save_header_to_np_array(self.stats)
        if self.dv_processing is not None and len(self.dv_processing):
            kwargs.update({
                'freq_min': self.dv_processing['freq_min'],
                'freq_max': self.dv_processing['freq_max'],
                'tw_len': self.dv_processing['tw_len'],
                'tw_start': self.dv_processing['tw_start'],
                'sides': self.dv_processing.get('sides', 'unknown'),
                'aligned': self.dv_processing.get('aligned', False)
            })
        kwargs.update({
            'method_array': np.array([self.method]),
            'vt_array': np.array([self.value_type]),
            'corr': self.corr,
            'value': self.value,
            'sim_mat': self.sim_mat,
            'second_axis': self.second_axis,
            'stretches': self.stretches,
            'corrs': self.corrs,
            'n_stat': self.n_stat
        })
        # Np load will otherwise be annoying
        to_save = {k: v for k, v in kwargs.items() if v is not None}
        np.savez_compressed(
            path, **to_save)

    def plot(
        self, save_dir: str = '.', figure_file_name: str = None,
        mark_time: datetime = None, normalize_simmat: bool = False,
        sim_mat_Clim: List[float] = [], xlim: Tuple[datetime, datetime] = None,
        ylim: Tuple[int, int] = None, plot_scatter: bool = False,
        figsize: Tuple[float, float] = (9, 11), dpi: int = 144,
        title: str = None, return_ax=False, style: str = 'technical',
        dateformat: str = '%d %b %y', ax: plt.Axes = None) -> Tuple[
            plt.figure, List[plt.axis]]:
        r"""
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
        :param xlim: Time Limits to plot, defaults to None.
        :type xlim: Tuple[datetime, datetime], optional
        :param ylim: Y-limits (e.g., stretch) of the plot, defaults to None
        :type ylim: Tuple[int, int], optional
        :param plot_scatter: If set to True, the scatter of all used
            combinations is plotted in form of a histogram.
            Defaults to False.
        :type plot_scatter: bool, optional
        :param figsize: Size of the figure/canvas, defaults to (9, 11)
        :type figsize: Tuple[float, float], optional
        :param dpi: Pixels per inch, defaults to 144
        :type dpi: int, optional
        :param title: Define a custom title. Otherwise, an automatic title
            will be created, defaults to None
        :type title: str, optional
        :param return_ax: Return plt.figure and list of axes.
            Defaults to False.
            This overwrites any choice to save the figure.
        :type return_ax: bool, optional
        :param style: Style of the plot. Defaults to 'technical'. The other
            option would be 'publication' (looks better but is less
            informative).
        :type style: str, optional
        :param dateformat: Format of the date on the x-axis. Defaults to
            '%d %b %y'.
        :type dateformat: str, optional
        :returns: If `return_ax` is set to True it returns fig and axes.
        """
        return plot_dv(
            style, self, save_dir=save_dir,
            figure_file_name=figure_file_name, mark_time=mark_time,
            normalize_simmat=normalize_simmat, sim_mat_Clim=sim_mat_Clim,
            figsize=figsize, dpi=dpi, xlim=xlim, ylim=ylim,
            title=title, plot_scatter=plot_scatter, return_ax=return_ax,
            dateformat=dateformat, ax=ax)

    def smooth_sim_mat(
            self, win_len: int,
            exclude_corr_below: Optional[float] = None,
            limit_times_to: Optional[
                Tuple[Tuple[int, int, float], Tuple[int, int, float]]] = None,
            limit_times_to_exclude: Optional[bool] = False):
        """
        Computes new dv/v and coherence values based on a smoothed
        version of the similarity matrix. The similarity matrix itself
        remains unaltered, so that no information is lost.

        :param win_len: Length of the window in number of samples.
        :type win_len: int
        :param exclude_corr_below: Exclude data points with a correlation
            coefficient below the chosen threshold. Defaults to None (i.e.,
            include all)
        :type exclude_corr_below: float
        :param limit_times_to: Limit the data to certain times of the day
            (e.g., (8, 0, 0), (16, 0, 0) for 8:00 to 16:00). Defaults to None
            (i.e., include all).
        :type limit_times_to: Tuple[Tuple[int, int, float],
            Tuple[int, int, float]]
        :param limit_times_to_exclude: Exclude the data points within the
            chosen time frame. Defaults to False.
        :type limit_times_to_exclude: bool

        :note::

            self.value and self.corr can always be repicked to obtain the
            original (unsmoothed) values.
        """
        if exclude_corr_below is not None or limit_times_to is not None:
            # make sure that data is not overwritten
            sim_mat = deepcopy(self.sim_mat)
        else:
            sim_mat = self.sim_mat
        if exclude_corr_below is not None:
            sim_mat[self.sim_mat < exclude_corr_below] = np.nan
        if limit_times_to is not None:
            start = time(*limit_times_to[0])
            end = time(*limit_times_to[1])
            corr_starts = np.array(
                [st.time for st in self.stats.corr_start])
            corr_ends = np.array(
                [st.time for st in self.stats.corr_end])
            mask = np.all(
                (corr_starts >= start, corr_ends <= end), axis=0)
            if limit_times_to_exclude:
                mask = ~mask
            sim_mat[~mask, :] = np.nan

        sim_mat = mu.nan_moving_av(sim_mat, int(win_len/2), axis=0)

        # Compute the dependencies again
        self.corr = np.nanmax(sim_mat, axis=1)
        self.value = self.second_axis[
            np.argmax(np.nan_to_num(sim_mat), axis=1)]
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
            except BadZipFile:
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
        dv_processing = dict(
            freq_min=float(loaded['freq_min']),
            freq_max=float(loaded['freq_max']),
            tw_start=float(loaded['tw_start']),
            tw_len=float(loaded['tw_len']))
        dv_processing['sides'] = loaded.get('sides', 'unknown')
        dv_processing['aligned'] = loaded.get('aligned', False)
    except KeyError:
        dv_processing = {}
    return DV(
        loaded['corr'], loaded['value'], vt, loaded['sim_mat'],
        loaded['second_axis'], method, stats=stats,
        stretches=loaded.get('stretches', None),
        corrs=loaded.get('corrs', None), n_stat=loaded.get('n_stat', None),
        dv_processing=dv_processing)
