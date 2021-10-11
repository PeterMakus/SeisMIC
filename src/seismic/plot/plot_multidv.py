'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3\
        <https://www.gnu.org/copyleft/lesser.html>_`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 5th October 2021 11:50:22 am
Last Modified: Monday, 11th October 2021 02:54:25 pm
'''

import os
from glob import glob
from typing import Tuple
from datetime import datetime

import matplotlib as mpl
from matplotlib import pyplot as plt

from seismic.monitor.dv import read_dv
from seismic.plot.plot_utils import set_mpl_params


def plot_multiple_dv(
    indir: str, only_mean: bool = False, title: str = None,
    outfile: str = None, fmt: str = None, dpi: int = 300, legend: bool = False,
    ylim: Tuple[float, float] = None,
        xlim: Tuple[datetime, datetime] = None):
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
    :param ylim: y limit, defaults to None
    :type ylim: Tuple[float, float], optional
    :param xlim: x limit, defaults to None
    :type xlim: Tuple[datetime, datetime], optional
    """
    set_mpl_params()
    if only_mean:
        pat = os.path.join(indir, '*.av-*.npz')
    else:
        pat = os.path.join(indir, '*.npz')
    statcodes = []
    for fi in glob(pat):
        dv = read_dv(fi)
        rtime = [utcdt.datetime for utcdt in dv.stats['corr_start']]
        plt.plot(rtime, -dv.value, '.', markersize=2)
        ax = plt.gca()
        statcodes.append(dv.stats.station)
    ax.xaxis.set_major_locator(mpl.dates.AutoDateLocator())

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
