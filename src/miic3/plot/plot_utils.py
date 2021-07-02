'''
:copyright:

:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 17th May 2021 12:25:54 pm
Last Modified: Friday, 2nd July 2021 11:57:22 am
'''

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def set_mpl_params():
    params = {
        # 'font.family': 'Avenir Next',
        'pdf.fonttype': 42,
        'font.weight': 'bold',
        'figure.dpi': 150,
        'axes.labelweight': 'bold',
        'axes.linewidth': .5,
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 13,
        'xtick.direction': 'in',
        'xtick.top': True,  # draw label on the top
        'xtick.bottom': True,  # draw label on the bottom
        'xtick.minor.visible': True,
        'xtick.major.top': True,  # draw x axis top major ticks
        'xtick.major.bottom': True,  # draw x axis bottom major ticks
        'xtick.minor.top': True,  # draw x axis top minor ticks
        'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'ytick.labelsize': 13,
        'ytick.direction': 'in',
        'ytick.left': True,  # draw label on the top
        'ytick.right': True,  # draw label on the bottom
        'ytick.minor.visible': True,
        'ytick.major.left': True,  # draw x axis top major ticks
        'ytick.major.right': True,  # draw x axis bottom major ticks
        'ytick.minor.left': True,  # draw x axis top minor ticks
        'ytick.minor.right': True,  # draw x axis bottom minor ticks
        'legend.fancybox': False,
        'legend.frameon': False,
        'legend.loc': 'upper left',
        'legend.numpoints': 2,
        'legend.fontsize': 'large',
        'legend.framealpha': 1,
        'legend.scatterpoints': 3,
        'legend.edgecolor': 'inherit'
    }
    matplotlib.rcParams.update(params)


def remove_all(ax=None, top=False, bottom=False, left=False, right=False,
               xticks='none', yticks='none'):
    """Removes all frames and ticks."""
    # Get current axis if none given.
    if ax is None:
        ax = plt.gca()

    # Hide the right and top spines
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)
    ax.spines['right'].set_visible(right)
    ax.spines['top'].set_visible(top)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position(yticks)
    ax.xaxis.set_ticks_position(xticks)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def remove_topright(ax=None):
    """Removes top and right border and ticks from input axes."""

    # Get current axis if none given.
    if ax is None:
        ax = plt.gca()

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_correlation(
    corr, tlim: list or tuple = None, ax: plt.Axes = None,
        outputdir: str = None, clean: bool = False) -> plt.Axes:
    set_mpl_params()

    # Get figure/axes dimensions
    if ax is None:
        width, height = 10, 2.5
        fig = plt.figure(figsize=(width, height))
        ax = plt.gca(zorder=9999999)
        axtmp = None
    else:
        bbox = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        axtmp = ax

    # The ratio ensures that the text
    # is perfectly distanced from top left/right corner
    ratio = width/height
    times = np.arange(
        corr.stats.start_lag, corr.stats.end_lag, corr.stats.delta)
    ydata = corr.data

    # Plot stuff into axes
    ax.fill_between(times, 0, ydata, where=ydata > 0,
                    interpolate=True, color=(0.9, 0.2, 0.2))
    ax.fill_between(times, 0, ydata, where=ydata < 0,
                    interpolate=True, color=(0.2, 0.2, 0.7))
    ax.plot(times, ydata, 'k', lw=0.75)

    # Set limits
    if tlim is None:
        ax.set_xlim(times[0], times[-1])
    else:
        ax.set_xlim(tlim)

    # Removes top/right axes spines. If you want the whole thing, comment
    # or remove
    remove_topright()

    # Plot RF only
    if clean:
        remove_all()
    else:
        ax.set_xlabel("Lag Time [s]")
        ax.set_ylabel("A    ", rotation=0)
        # Start time in station stack does not make sense
        text = corr.stats.corr_start.isoformat(sep=" ") + "\n" + corr.get_id()
        ax.text(0.995, 1.0-0.005*ratio, text, transform=ax.transAxes,
                horizontalalignment="right", verticalalignment="top")

    # Only use tight layout if not part of plot.
    if axtmp is None:
        plt.tight_layout()

    # Outout the receiver function as pdf using
    # its station name and starttime
    if outputdir is not None:
        filename = os.path.join(
            outputdir, rf.get_id() + "_"
            + rf.stats.starttime._strftime_replacement('%Y%m%dT%H%M%S')
            + ".pdf")
        plt.savefig(filename, format="pdf")
    return ax
