'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 19th July 2021 11:37:54 am
Last Modified: Tuesday, 28th February 2023 10:51:04 am
'''
import os
from typing import Tuple, Optional, List
import datetime

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
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
            corr.stats.corr_start.strftime('%Y/%m/%d %H:%M'),
            corr.stats.corr_end.strftime('%Y/%m/%d %H:%M'),
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
            + corr.stats.starttime.strftime('%Y%m%dT%H%M%S')
            + ".pdf")
        plt.savefig(filename, format="pdf")
    return ax


def plot_cst(
    cst: Stream, sort_by: str = 'corr_start',
    timelimits: list or tuple or None = None,
    ylimits: list or tuple or None = None, scalingfactor: float = 2.0,
    ax: plt.Axes = None, linewidth: float = 0.25,
    outputfile: str or None = None, title: str or None = None,
    type: str = 'heatmap', cmap: str = 'seismic', vmin: float = None,
        vmax: float = None, **kwargs):
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
    :type type: str, optional
    :param cmap: Decides about colormap if type == 'heatmap'.
        Defaults to 'inferno'.
    :type cmap: str, optional
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
            times = heat_plot_corr_start(
                cst, ax, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            raise NotImplementedError(
                'Unknown or not implemented plot type %s.' % type)
    elif sort_by == 'dist':
        if type == 'section':
            scalingfactor *= 4
            times = sect_plot_dist(
                cst, ax, scalingfactor, linewidth, **kwargs)
        elif type == 'heatmap':
            times = heat_plot_corr_dist(
                cst, ax, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
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

    plt.xlabel(r"$\tau$ [s]")

    plt.title(title)

    # Set output directory
    if outputfile is not None:
        plt.savefig(outputfile, dpi=300, transparent=True)
    return ax


def plot_corr_bulk(
    corr_bulk,
    timelimits: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
    ylimits: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
    clim: Optional[Tuple[float, float]] = None,
    plot_colorbar: bool = False, outputfile: Optional[str] = None,
    title: Optional[str] = None,
        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plots a :class:`~seismic.correlate.stream.CorrBulk` object.

    :param corr_bulk: The CorrBulk to plot.
    :type corr_bulk: :class:`~seismic.correlate.stream.CorrBulk`
    :param timelimits: Limits time axis, defaults to None
    :type timelimits: Optional[Tuple[datetime.datetime, datetime.datetime]],
        optional
    :param ylimits: Limits of y-axis, defaults to None
    :type ylimits: Optional[Tuple[datetime.datetime, datetime.datetime]],
        optional
    :param clim: Limits of Colobar, defaults to None
    :type clim: Optional[Tuple[float, float]], optional
    :param plot_colorbar: add colorbar to plot, defaults to False
    :type plot_colorbar: bool, optional
    :param outputfile: save file to, defaults to None
    :type outputfile: Optional[str], optional
    :param title: Title, defaults to None
    :type title: Optional[str], optional
    :param ax: Axis to plot into, defaults to None
    :type ax: Optional[plt.Axes], optional
    :return: The current axis
    :rtype: plt.Axes
    """
    set_mpl_params()

    # Create figure if no axes is specified
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.axes()  # zorder=9999999

    extent = [
        corr_bulk.stats.start_lag,
        corr_bulk.stats.end_lag,
        date2num(corr_bulk.stats.corr_start[0].datetime),
        date2num(corr_bulk.stats.corr_start[-1].datetime)]
    im = ax.imshow(corr_bulk.data, extent=extent, aspect='auto')
    ax.yaxis_date()
    ax.figure.autofmt_xdate(rotation=45)
    ax.set_xlabel(r'$\{tau}$ [s]')

    # Set limits
    if ylimits:
        ax.set_ylim(date2num(ylimits[0]), date2num(ylimits[1]))
    if timelimits:
        ax.set_xlim(timelimits)
    if title:
        ax.set_title(title)
    if clim:
        im.set_clim(clim)
    if plot_colorbar:
        plt.colorbar(im, orientation='vertical')
    if outputfile:
        plt.savefig(outputfile)
    else:
        plt.show()
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

    ax.yaxis.set_major_formatter(mpl.dates.DateFormatter('%d %h %y'))
    ax.invert_yaxis()
    return times


def heat_plot_corr_start(
        cst: Stream, ax: plt.Axes, cmap: str, vmin: float, vmax: float):
    data = np.empty((cst.count(), cst[0].stats.npts))
    # y grid
    y = []
    for ii, ctr in enumerate(cst):
        data[ii, :] = ctr.data
        y.append(ctr.stats['corr_start'].datetime)
        times = ctr.times()
    ds = plt.pcolormesh(
        times, np.array(y), data, shading='auto', cmap=cmap, vmin=vmin,
        vmax=vmax)
    plt.colorbar(
        ds, label='correlation coefficient', shrink=.6,
        orientation='horizontal')
    ax.yaxis.set_major_locator(mpl.dates.AutoDateLocator())
    ax.yaxis.set_major_formatter(mpl.dates.DateFormatter('%d %h %y'))
    ax.invert_yaxis()
    return times


def sect_plot_dist(
    cst: Stream, ax: plt.Axes, scalingfactor: float,
    linewidth: float, plot_reference_v: bool = False,
        ref_v: List[float] = [1, 2, 3]) -> np.ndarray:
    for ii, ctr in enumerate(cst):
        ydata = ctr.data
        times = ctr.times()

        ytmp = ydata * scalingfactor + ctr.stats.dist

        ax.plot(times, ytmp, 'k', lw=linewidth, zorder=-ii + 0.1)
    if plot_reference_v:
        yref = np.array([ax.get_ylim()[1], 0, ax.get_ylim()[1]])
        [plt.plot(
            [-yref[0]/rv, 0, yref[0]/rv], yref, linewedith=.5, zorder=0,
            label=f'{rv}'+r'$\frac{km}{s}$') for rv in ref_v]
        plt.legend()
    plt.ylabel(r"Distance [km]")
    # Set label locations.
    # step = round((cst[-1].stats.dist - cst[0].stats.dist)/10000)
    # plt.yticks(np.arange(0, ctr.stats.dist/1000+step, step, dtype=int))
    plt.yticks(np.linspace(0, ctr.stats.dist, 10, dtype=int))
    return times


def heat_plot_corr_dist(
    cst: Stream, ax: plt.Axes, cmap: str, vmin: float, vmax: float,
        plot_reference_v: bool = False, ref_v: List[float] = [1, 2, 3]):
    data = np.empty((cst.count(), cst[0].stats.npts))
    # y grid
    y = np.zeros((len(cst),))
    # x coords
    times = cst[0].times()
    for ii, ctr in enumerate(cst):
        data[ii, :] = ctr.data
        y[ii] = ctr.stats.dist
    ds = plt.pcolormesh(
        times, y, data, shading='auto', cmap=cmap, vmin=vmin,
        vmax=vmax)
    if plot_reference_v:
        yref = np.array([ax.get_ylim()[1], 0, ax.get_ylim()[1]])
        [plt.plot(
            [-yref[0]/rv, 0, yref[0]/rv], yref, linewedith=.5, zorder=0,
            label=f'{rv}'+r'$\frac{km}{s}$') for rv in ref_v]
        plt.legend()
    plt.colorbar(
        ds, label='correlation coefficient', shrink=.6,
        orientation='horizontal')
    plt.yticks(np.linspace(0, ctr.stats.dist, 10, dtype=int))
    ax.set_ylabel(r"Distance [km]")
    return times
