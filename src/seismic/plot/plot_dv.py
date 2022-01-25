'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 16th July 2021 02:30:02 pm
Last Modified: Wednesday, 19th January 2022 02:44:28 pm
'''

from typing import Tuple
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os

from seismic.plot.plot_utils import set_mpl_params


def plot_dv(
    dv, save_dir='.', figure_file_name=None, mark_time=None,
    normalize_simmat=False, sim_mat_Clim=[], figsize=(9, 11), dpi=72,
        ylim: Tuple[float, float] = None, title: str = None):
    """ Plot the "extended" dv dictionary

    This function is thought to plot the result of the velocity change estimate
    as output by :class:`~miic.core.stretch_mod.multi_ref_vchange_and_align`
    and successively "extended" to contain also the timing in the form
    {'time': time_vect} where `time_vect` is a :class:`~numpy.ndarray` of
    :class:`~datetime.datetime` objects.
    This addition can be done, for example, using the function
    :class:`~miic.core.miic_utils.add_var_to_dict`.
    The produced figure is saved in `save_dir` that, if necessary, it is
    created.
    It is also possible to pass a "special" time value `mark_time` that will be
    represented in the `dv/v` and `corr` plot as a vertical line; It can be
    a string (i.e. YYYY-MM-DD) or directly a :class:`~datetime.datetime`
    object.
    if the `dv` dictionary also contains a 'comb_mseedid' keyword, its `value`
    (i.e. MUST be a string) will be reported in the title.
    In case of the chosen filename exist in the `save_dir` directory, a prefix
    _<n> with n:0..+Inf, is added.
    The aspect of the plot may change depending on the matplotlib version. The
    recommended one is matplotlib 1.1.1

    :type dv: dict
    :param dv: velocity change estimate dictionary as output by
        :class:`~miic.core.stretch_mod.multi_ref_vchange_and_align` and
        successively "extended" to contain also the timing in the form
        {'time': time_vect} where `time_vect` is a :class:`~numpy.ndarray` of
        :class:`~datetime.datetime` objects.
    :type save_dir: string
    :param save_dir: Directory where to save the produced figure. It is created
        if it doesn't exist.
    :type figure_file_name: string
    :param figure_file_name: filename to use for saving the figure. If None
        the figure is displayed in interactive mode.
    :type mark_time: string or :class:`~datetime.datetime` object
    :param mark_time: It is a "special" time location that will be represented
        in the `dv/v` and `corr` plot as a vertical line.
    :type normalize_simmat: Bool
    :param normalize_simmat: if True the simmat will be normalized to a maximum
        of 1. Defaults to False
    :type sim_mat_Clim: 2 element array_like
    :param sim_mat_Clim: if non-empty it set the color scale limits of the
        similarity matrix image
    :param ylim: Limits for the stretch axis. Defaults to None
    :type ylim: Tuple[float, float], optional
    """
    set_mpl_params()

    if sim_mat_Clim and len(sim_mat_Clim) != 2:
        raise ValueError('Sim_mat_Clim has to be a list or Tuple of length 2.')

    if figure_file_name:
        os.makedirs(save_dir, exist_ok=True)

    # Create a unique filename if TraitsUI-default is given
    if figure_file_name == 'plot_default':
        fname = figure_file_name + '_change.png'
        exist = os.path.isfile(os.path.join(save_dir, fname))
        i = 0
        while exist:
            fname = "%s_%i" % (figure_file_name, i)
            exist = os.path.isfile(os.path.join(save_dir,
                                                fname + '_change.png'))
            i += 1
        figure_file_name = fname

    # Extract the data from the dictionary

    value_type = dv['value_type']
    method = dv['method']

    corr = dv['corr']
    dt = dv['value']
    sim_mat = dv['sim_mat']
    stretch_vect = dv['second_axis']

    rtime = [utcdt.datetime for utcdt in dv['stats']['corr_start']]

    # normalize simmat if requested
    if normalize_simmat:
        sim_mat = sim_mat/np.tile(
            np.max(sim_mat, axis=1), (sim_mat.shape[1], 1)).T

    stretching_amount = np.max(stretch_vect)

    # Adapt plot details in agreement with the type of dictionary that
    # has been passed

    if (value_type == 'stretch') and (method == 'single_ref'):

        tit = "Single reference dv/v"
        dv_tick_delta = round(stretch_vect.max()/5, 2)  # 0.01
        dv_y_label = "dv/v"
        # plotting velocity requires to flip the stretching axis
    elif (value_type == 'stretch') and (method == 'multi_ref'):

        tit = "Multi reference dv/v"
        dv_tick_delta = round(stretch_vect.max()/5, 2)  # 0.01
        dv_y_label = "dv/v"
        # plotting velocity requires to flip the stretching axis
    elif (value_type == 'shift') and (method == 'time_shift'):

        tit = "Time shift"
        dv_tick_delta = 5
        dv_y_label = "time shift (sample)"

    else:
        raise ValueError(f"Unknown dv type, {value_type}!")

    f = plt.figure(figsize=figsize, dpi=dpi)

    gs = mpl.gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])

    ax1 = f.add_subplot(gs[0])
    imh = plt.imshow(
        np.flipud(sim_mat.T).astype(float), interpolation='none',
        aspect='auto')

    # plotting value is way easier now
    plt.plot(-dv['value'], 'b.')
    # Set extent so we can treat the axes properly (mainly y)
    imh.set_extent((0, sim_mat.shape[0], stretch_vect[-1], stretch_vect[0]))

    ###
    ax1.set_xlim(0, sim_mat.shape[0])
    plt.xlim(0, sim_mat.shape[0])

    if value_type == 'stretch':
        ax1.invert_yaxis()
    if ylim:
        plt.ylim(ylim)
    # ###
    if sim_mat_Clim:
        imh.set_clim(sim_mat_Clim[0], sim_mat_Clim[1])

    plt.gca().get_xaxis().set_visible(False)

    stats = dv['stats']
    comb_mseedid = '%s.%s.%s.%s' % (
        stats['network'], stats['station'], stats['location'],
        stats['channel'])
    if title:
        tit = title
    else:
        tit = "%s estimate (%s)" % (tit, comb_mseedid)

    ax1.set_title(tit)
    ax1.yaxis.set_ticks_position('right')
    ax1.yaxis.set_label_position('left')
    # ax1.yaxis.label.set_rotation(270)
    # ax1.yaxis.set_label_coords(1.03, 0.5)
    ax1.set_xticklabels([])
    ax1.set_ylabel(dv_y_label)

    ax2 = f.add_subplot(gs[1])
    plt.plot(rtime, -dt, '.')
    if 'model_value' in dv.keys():
        plt.plot(rtime, -dv['model_value'], 'g.')
    plt.xlim([rtime[0], rtime[-1]])
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim((-stretching_amount, stretching_amount))
    if mark_time and not (
            np.all(rtime < mark_time) and np.all(rtime > mark_time)):
        plt.axvline(mark_time, lw=1, color='r')
    ax2.yaxis.set_ticks_position('left')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.label.set_rotation(270)
    ax2.yaxis.set_label_coords(1.03, 0.5)
    ax2.set_ylabel(dv_y_label)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(dv_tick_delta))
    ax2.yaxis.grid(True, 'major', linewidth=1)
    ax2.xaxis.grid(True, 'major', linewidth=1)
    ax2.set_xticklabels([])

    ax3 = f.add_subplot(gs[2])
    plt.plot(rtime, corr, '.')
    if 'model_corr' in dv.keys():
        plt.plot(rtime, dv['model_corr'], 'g.')
    plt.xlim([rtime[0], rtime[-1]])
    ax3.yaxis.set_ticks_position('right')
    ax3.set_ylabel("Correlation")
    plt.ylim((0, 1))
    if mark_time and not (
            np.all(rtime < mark_time) and np.all(rtime > mark_time)):
        plt.axvline(mark_time, lw=1, color='r')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax3.yaxis.grid(True, 'major', linewidth=1)
    ax3.xaxis.grid(True, 'major', linewidth=1)

    plt.subplots_adjust(hspace=0, wspace=0)

    if figure_file_name is None:
        plt.show()
    else:
        print('saving to %s' % figure_file_name)
        f.savefig(os.path.join(save_dir, figure_file_name + '_change.png'),
                  dpi=dpi)
        plt.close()
