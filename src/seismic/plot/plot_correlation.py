'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 19th July 2021 11:37:54 am
Last Modified: Friday, 17th December 2021 01:51:43 pm
'''
import os
import warnings

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from obspy.core.utcdatetime import UTCDateTime
from obspy import Trace, Stream

from seismic.plot.plot_utils import set_mpl_params, remove_topright, remove_all


def plot_ctr(
    corr: Trace, tlim: list or tuple = None, ax: plt.Axes = None,
        outputdir: str = None, clean: bool = False) -> plt.Axes:
    set_mpl_params()

    # Get figure/axes dimensions
    if ax is None:
        width, height = 10, 2.5
        fig = plt.figure(figsize=(width, height))
        ax = plt.gca()
        axtmp = None
    else:
        fig = plt.gcf()
        bbox = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        axtmp = ax

    # The ratio ensures that the text
    # is perfectly distanced from top left/right corner
    ratio = width/height
    times = corr.times()
    ydata = corr.data

    # Plot stuff into axes
    ax.plot(times, ydata, 'k', lw=1)

    # Set limits
    if tlim is None:
        ax.set_xlim(times[0], times[-1])
    else:
        ax.set_xlim(tlim)

    ax.set_ylim([-1, 1])

    # Removes top/right axes spines. If you want the whole thing, comment
    # or remove
    remove_topright()

    # Plot RF only
    if clean:
        remove_all()
    else:
        ax.set_xlabel("Lag Time [s]")
        ax.set_ylabel("Correlation")
        # Start time in station stack does not make sense
        text = '%s - %s\n%s' % (
            corr.stats.corr_start._strftime_replacement('%Y/%m/%d %H:%M'),
            corr.stats.corr_end._strftime_replacement('%Y/%m/%d %H:%M'),
            corr.get_id())
        # corr.stats.corr_start.isoformat(sep=" ") + "\n" + corr.get_id()
        ax.text(0.995, 1.0-0.005*ratio, text, transform=ax.transAxes,
                horizontalalignment="right", verticalalignment="top")

    # Only use tight layout if not part of plot.
    if axtmp is None:
        plt.tight_layout()

    # Hide minor ticks on y-axis
    plt.tick_params(axis='y', which='minor', length=0)

    # Outout the receiver function as pdf using
    # its station name and starttime
    if outputdir is not None:
        filename = os.path.join(
            outputdir, corr.get_id() + "_"
            + corr.stats.starttime._strftime_replacement('%Y%m%dT%H%M%S')
            + ".pdf")
        plt.savefig(filename, format="pdf")
    return ax


def plot_cst(
    cst: Stream, sort_by: str = 'corr_start',
    timelimits: list or tuple or None = None,
    ylimits: list or tuple or None = None, scalingfactor: float = 2.0,
    ax: plt.Axes = None, linewidth: float = 0.25,
    outputfile: str or None = None, title: str or None = None,
        type: str = 'heatmap'):
    """
    Creates a section or heat plot of all correlations in this stream.

    :param cst: Input CorrelationStream
    :type cst: :class:`~seismic.correlate.stream.CorrStream`
    :param sort_by: Which parameter to plot against. Can be either
        ``corr_start`` or ``distance``, defaults to 'corr_start'.
    :type sort_by: str, optional
    :param timelimits: xlimits (lag time) in seconds, defaults to None
    :type timelimits: Tuple[float, float], optional
    :param ylimits: limits for Y-axis (either a :class:`datetime.datetime`
        or float in km (if plotted against distance)), defaults to None.
    :type ylimits: Tuple[float, float], optional
    :param scalingfactor: Which factor to scale the Correlations with. Play
        around with this if you want to make amplitudes bigger or smaller,
        defaults to None (automatically chosen).
    :type scalingfactor: float, optional
    :param ax: Plot in existing axes? Defaults to None
    :type ax: plt.Axes, optional
    :param linewidth: Width of the lines to plot, defaults to 0.25
    :type linewidth: float, optional
    :param outputfile: Save the plot? defaults to None
    :type outputfile: str, optional
    :param title: Title of the plot, defaults to None
    :type title: str, optional
    :param type: can be set to 'heat' for a heatmap or 'section' for
        a wiggle type plot. Default to `heat`
    :type type: str
    :return: returns the axes object.
    :rtype: `matplotlib.pyplot.Axes`

    """
    set_mpl_params()

    scalingfactor = scalingfactor or 1

    # Create figure if no axes is specified
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.axes()  # zorder=9999999
    if sort_by == 'distance':
        sort_by = 'dist'
    cst.sort(keys=[sort_by, 'network', 'station', 'channel'])

    if not cst.count():
        raise KeyError('No Correlations with the desired properties found.')

    # Plot traces
    if sort_by == 'corr_start':
        scalingfactor *= 1e5
        if type == 'section':
            times = sect_plot_corr_start(cst, ax, scalingfactor, linewidth)
        elif type == 'heatmap':
            times = heat_plot_corr_start(cst, ax)
        else:
            raise NotImplementedError(
                'Unknown or not implemented plot type %s.' % type)
    elif sort_by == 'dist':
        if type != 'section':
            warnings.warn(
                'Distance plot only supports section type.'
            )
        scalingfactor *= 4
        times = sect_plot_dist(cst, ax, scalingfactor, linewidth)
    else:
        raise NotImplementedError('Unknown sorting method %s.' % sort_by)

    plt.tick_params(axis='y', which='minor', length=0)

    # Set limits
    if ylimits:
        plt.ylim(ylimits)

    if timelimits is None:
        plt.xlim(times[0], times[-1])
    else:
        plt.xlim(timelimits)

    plt.xlabel(r"Lag Time [s]")

    plt.title(title)

    # Set output directory
    if outputfile is None:
        plt.show()
    else:
        plt.savefig(outputfile, dpi=300, transparent=True)
    return ax


def sect_plot_corr_start(
    cst: Stream, ax: plt.Axes, scalingfactor: float,
        linewidth: float) -> np.ndarray:
    for ii, ctr in enumerate(cst):
        ydata = ctr.data
        times = ctr.times()

        ytmp = ctr.stats['corr_start'].timestamp

        ctmp = ydata * scalingfactor + ytmp
        ctmp = np.array([UTCDateTime(t).datetime for t in ctmp])

        ax.plot(times, ctmp, 'k', lw=linewidth, zorder=-ii + 0.1)

    ax.yaxis.set_major_locator(mpl.dates.AutoDateLocator())

    ax.yaxis.set_major_formatter(mpl.dates.DateFormatter('%d %h'))
    return times


def heat_plot_corr_start(cst: Stream, ax: plt.Axes):
    data = np.empty((cst.count(), cst[0].stats.npts))
    # y grid
    y = []
    for ii, ctr in enumerate(cst):
        data[ii, :] = ctr.data
        y.append(ctr.stats['corr_start'].datetime)
        times = ctr.times()
    ds = plt.pcolormesh(times, np.array(y), data, shading='auto')
    plt.colorbar(ds)
    ax.yaxis.set_major_locator(mpl.dates.AutoDateLocator())
    ax.yaxis.set_major_formatter(mpl.dates.DateFormatter('%d %h'))
    return times


def sect_plot_dist(
    cst: Stream, ax: plt.Axes, scalingfactor: float,
        linewidth: float) -> np.ndarray:
    for ii, ctr in enumerate(cst):
        ydata = ctr.data
        times = ctr.times()

        ytmp = ydata * scalingfactor + ctr.stats.dist

        ax.plot(times, ytmp, 'k', lw=linewidth, zorder=-ii + 0.1)
        plt.ylabel(r"Distance [km]")
    # Set label locations.
    # step = round((cst[-1].stats.dist - cst[0].stats.dist)/10000)
    # plt.yticks(np.arange(0, ctr.stats.dist/1000+step, step, dtype=int))
    plt.yticks(np.linspace(0, ctr.stats.dist, 10, dtype=int))
    return times
