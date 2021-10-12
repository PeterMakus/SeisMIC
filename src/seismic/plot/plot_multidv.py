'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3\
        <https://www.gnu.org/copyleft/lesser.html>_`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 5th October 2021 11:50:22 am
Last Modified: Tuesday, 12th October 2021 12:09:52 pm
'''

from logging import warn
import os
import fnmatch
from glob import glob
from typing import List, Tuple
from datetime import datetime

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

from seismic.monitor.dv import read_dv
from seismic.plot.plot_utils import set_mpl_params


def plot_multiple_dv(
    indir: str, only_mean: bool = False, title: str = None,
    outfile: str = None, fmt: str = None, dpi: int = 300, legend: bool = False,
    plot_median: bool = False, ylim: Tuple[float, float] = None,
        xlim: Tuple[datetime, datetime] = None, statfilter: List[str] = None):
    """
    Plots several Velocity variations in one single plot

    :param indir: directory in which the .npz DV files are located.
    :type indir: str
    :param only_mean: plot only the average of one station instead of all
        of its components. **The averages have to be computed before**.
        Defaults to false.
    :type only_mean: bool, optional
    :param title: Title for the figure, defaults to None
    :type title: str, optional
    :param outfile: File to save figure to, defaults to None
    :type outfile: str, optional
    :param fmt: Figure format, defaults to None
    :type fmt: str, optional
    :param dpi: DPI for pixel-based formats, defaults to 300
    :type dpi: int, optional
    :param legend: Should a legend be plotted, defaults to False
    :type legend: bool, optional
    :param plot_median: Plot the median of all dataset as a black line.
        Defaults to False.
    :type plot_median: bool, optional.
    :param ylim: y limit, defaults to None
    :type ylim: Tuple[float, float], optional
    :param xlim: x limit, defaults to None
    :type xlim: Tuple[datetime, datetime], optional
    :param statfilter: Only plot data from the station combinations with the
        following codes. Given in the form
        ['net0-net0.sta0-stat1.ch0-ch1', ...]. Note that wildcards are allowed.
        Defaults to None.
    :type statfilter: List[str]
    """
    set_mpl_params()
    if only_mean:
        pat = os.path.join(indir, '*.av-*.npz')
    else:
        pat = os.path.join(indir, '*.npz')
    statcodes = []
    vals = []
    infiles = glob(pat)
    if statfilter is not None:
        statfilter = ['*%s*' % filt for filt in statfilter]
        infiles_new = []
        for filt in statfilter:
            infiles_new.extend(fnmatch.filter(infiles, filt))
        infiles = infiles_new
    for fi in infiles:
        try:
            dv = read_dv(fi)
        except Exception:
            warn('Corrupt file %s discovered...skipping.' % fi)
        rtime = [utcdt.datetime for utcdt in dv.stats['corr_start']]
        plt.plot(rtime, -dv.value, '.', markersize=.25)
        vals.append(-dv.value)
        ax = plt.gca()
        statcodes.append(dv.stats.station)
    ax.xaxis.set_major_locator(mpl.dates.AutoDateLocator())
    if plot_median:
        mean = np.nanmedian(vals, axis=0)
        plt.plot(rtime, mean, 'k')
        statcodes.append('median')

    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %h'))
    plt.xticks(rotation='vertical')
    plt.ylabel('dv/v')
    if legend:
        plt.legend(statcodes)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    if outfile:
        plt.tight_layout()
        if fmt == 'pdf' or fmt == 'svg':
            plt.savefig('%s.%s' % (outfile, fmt), format=fmt)
        else:
            plt.savefig('%s.%s' % (outfile, fmt), format=fmt, dpi=dpi)
    else:
        plt.show()
