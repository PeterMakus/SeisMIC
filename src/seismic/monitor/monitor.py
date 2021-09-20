'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 3rd June 2021 04:15:57 pm
Last Modified: Monday, 20th September 2021 04:58:50 pm
'''
from copy import deepcopy
import logging
from math import inf
import os
from typing import List, Tuple
import yaml
import fnmatch
from glob import glob

from mpi4py import MPI
import numpy as np
from obspy import UTCDateTime
from tqdm import tqdm

from seismic.db.corr_hdf5 import CorrelationDataBase
from seismic.monitor.dv import DV, read_dv
from seismic.utils.miic_utils import log_lvl


class Monitor(object):
    def __init__(self, options: dict or str):
        """
        Object that handles the computation of seismic velocity changes.
        This will access correlations that have been computed previously with
        the :class:`~seismic.correlate.correlate.Correlator` class.

        Takes the parameter file as input (either as yaml or as dict).

        :param options: Parameters for the monitor. Usually, they come in form
            of a *yaml* file. If this parameter is a str, it will be
            interpreted to be the path to said *yaml* file.
        :type options: dict or str
        """
        if isinstance(options, str):
            with open(options) as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        self.options = options
        self.starttimes, self.endtimes = self._starttimes_list()
        self.proj_dir = options['proj_dir']
        self.outdir = os.path.join(
            options['proj_dir'], options['dv']['subdir'])
        self.indir = os.path.join(
            options['proj_dir'], options['co']['subdir']
        )

        # init MPI
        self.comm = MPI.COMM_WORLD
        self.psize = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # directories:
        logdir = os.path.join(self.proj_dir, options['log_subdir'])
        if self.rank == 0:
            os.makedirs(self.outdir, exist_ok=True)
            os.makedirs(logdir, exist_ok=True)

        # Logging - rank dependent
        loglvl = log_lvl[self.options['log_level']]

        if self.rank == 0:
            tstr = UTCDateTime.now().strftime('%Y-%m-%d-%H:%M')
        else:
            tstr = None
        tstr = self.comm.bcast(tstr, root=0)
        self.logger = logging.getLogger(
            "seismic.monitor.Monitor%s" % str(self.rank).zfill(3))
        self.logger.setLevel(loglvl)

        # also catch the warnings
        logging.captureWarnings(True)
        warnlog = logging.getLogger('py.warnings')
        fh = logging.FileHandler(os.path.join(logdir, 'monitor%srank%s' % (
            tstr, str(self.rank).zfill(3))))
        fh.setLevel(loglvl)
        self.logger.addHandler(fh)
        warnlog.addHandler(fh)
        fmt = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(fmt)
        self.logger.addHandler(consoleHandler)

        # Find available stations and network
        self.netlist, self.statlist, self.infiles = self._find_available_corrs(
            )

    def _starttimes_list(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns numpy array of starttimes and endtimes, for which each data
        point of velocity change should be computed.

        :return: A numpy array holding the starttimes and endtimes of the
            computing time windows in seconds (as UTC timestamps).
        :rtype: Tuple[np.ndarray, np.ndarray]

        ..seealso:: For complete description,
            :func:`~seismic.monitor.monitor.make_time_list`
        """
        starttimes, endtimes = make_time_list(
            self.options['dv']['start_date'], self.options['dv']['end_date'],
            self.options['dv']['date_inc'], self.options['dv']['win_len'])
        return starttimes, endtimes

    def _find_available_corrs(self) -> Tuple[List[str], List[str], List[str]]:
        """
        1. Finds the files of available correlations.
        2. Out of those files, it will extract the ones that are requested.

        :return: list of network combination codes, list of station combination
            codes, and list of the file paths to their corresponding
            correlation files.
        :rtype: Tuple[List[str], List[str], List[str]]
        """
        netlist, statlist, infiles = corr_find_filter(
            self.indir, **self.options)
        self.logger.info('Found correlation data for the following station \
and network combinations %s' % str(
                ['{n}.{s}'.format(n=n, s=s) for n, s in zip(
                    netlist, statlist)]))
        return netlist, statlist, infiles

    def compute_velocity_change(
        self, corr_file: str, tag: str, network: str, station: str,
            channel: str):
        """
        Computes the velocity change for a cross (or auto) correlation
        time-series. This process is executed "per station combination".
        Meaning, several processes of this method can be started via MPI
        and the method
        :meth:`~seismic.monitor.monitor.Monitor.compute_velocity_change_bulk`,
        which is the function that should rather be acessed than this one.
        Data will be written into the defined dv folder and plotted if this
        option was previously set to True.

        :param corr_file: File to read the correlation data from.
        :type corr_file: str
        :param tag: Tag of the data in the file (almost always `'subdivision'`)
        :type tag: str
        :param network: Network combination code
        :type network: str
        :param station: Station Combination Code
        :type station: str
        :param channel: Channel combination code
        :type channel: str
        """
        self.logger.info('Computing velocity change for file: %s and channel:\
%s' % (corr_file, channel))
        with CorrelationDataBase(corr_file, mode='r') as cdb:
            # get the corrstream containing all the corrdata for this combi
            cst = cdb.get_data(network, station, channel, tag)
        cb = cst.create_corr_bulk(inplace=True)

        # for possible rest bits
        del cst
        # Do the actual processing:
        cb.normalize(normtype='absmax')
        # That is were the stacking is happening
        cb.resample(self.starttimes, self.endtimes)
        cb.filter(
            (self.options['dv']['freq_min'], self.options['dv']['freq_max']))

        # Preprocessing on the correlation bulk
        for func in self.options['dv']['preprocessing']:
            f = cb.__getattribute__(func['function'])
            cb = f(**func['args'])

        # Now, we make a copy of the cm to be trimmed
        cbt = cb.copy().trim(
            -(self.options['dv']['tw_start']+self.options['dv']['tw_len']),
            (self.options['dv']['tw_start']+self.options['dv']['tw_len']))

        if cbt.data.shape[1] <= 20:
            raise ValueError('CorrBulk extremely short.')

        tr = cbt.extract_trace(method='mean')

        # Compute time window
        tw = [np.arange(
            self.options['dv']['tw_start']*cbt.stats['sampling_rate'],
            (self.options['dv']['tw_start']+self.options[
                'dv']['tw_len'])*cbt.stats['sampling_rate'], 1)]
        dv = cbt.stretch(
            ref_trc=tr, return_sim_mat=True,
            stretch_steps=self.options['dv']['stretch_steps'],
            stretch_range=self.options['dv']['stretch_range'],
            tw=tw)
        ccb = cb.correct_stretch(dv)

        ccb.trim(
            -(self.options['dv']['tw_start']+self.options['dv']['tw_len']),
            (self.options['dv']['tw_start']+self.options['dv']['tw_len']))

        # extract the final reference trace (mean excluding very different
        # traces)
        tr = ccb.extract_trace(method='mean')
        # obtain an improved time shift measurement
        dv = cbt.stretch(
            ref_trc=tr, return_sim_mat=True,
            stretch_steps=self.options['dv']['stretch_steps'],
            stretch_range=self.options['dv']['stretch_range'],
            tw=tw)
        outf = os.path.join(
            self.outdir, 'DV-%s.%s.%s' % (network, station, channel))
        dv.save(outf)
        if self.options['dv']['plot_vel_change']:
            fname = '%s_%s_%s' % (
                dv.stats.network, dv.stats.station, dv.stats.channel)
            savedir = os.path.join(
                self.options['proj_dir'], self.options['fig_subdir'])
            dv.plot(
                save_dir=savedir, figure_file_name=fname,
                normalize_simmat=True, sim_mat_Clim=[-1, 1])

    def compute_velocity_change_bulk(self):
        """
        Compute the velocity change for all correlations using MPI.
        The parameters for the correlation defined in the *yaml* file will be
        used.

        This function will just call
        :meth:`~seismic.monitor.monitor.Monitor.compute_velocity_change`
        several times.
        """
        tag = 'subdivision'
        # get number of available channel combis
        if self.rank == 0:
            plist = []
            for f, n, s in zip(self.infiles, self.netlist, self.statlist):
                with CorrelationDataBase(f, mode='r') as cdb:
                    ch = cdb.get_available_channels(
                        tag, n, s)
                    plist.extend([f, n, s, c] for c in ch)
        else:
            plist = None
        plist = self.comm.bcast(plist, root=0)
        pmap = (np.arange(len(plist))*self.psize)/len(plist)
        pmap = pmap.astype(np.int32)
        ind = pmap == self.rank
        ind = np.arange(len(plist), dtype=int)[ind]

        # Assign a task to each rank
        for ii in tqdm(ind):
            corr_file, net, stat, cha = plist[ii]
            try:
                self.compute_velocity_change(
                    corr_file, tag, net, stat, cha)
            except Exception as e:
                self.logger.error(e)

    def compute_components_average(self, method: str = 'AutoComponents'):
        """
        Averages the Similarity matrix of different velocity changes in
        the whole dv folder. Based upon those values, new dvs and correlations
        are computed.

        :param method: Which components should be averaged? Can be
            'StationWide' if all component-combinations should be averaged or
            'AutoComponents' if only the dv results of autocorrelation are to
            be averaged. Defaults to 'AutoComponents'
        :type method: str, optional
        :raises ValueError: For Unknown combination methods.
        """
        if method.lower() not in ('autocomponents', 'stationwide'):
            raise ValueError(
                'Unknown averaging method. ' +
                'Use "autocomponent" or "stationwide".'
            )
        infiles = glob(os.path.join(self.outdir, '*.npz'))

        while len(infiles):
            # Find files belonging to same station
            pat = '.'.join(infiles[0].split('.')[:-2]) + '*'
            filtfil = fnmatch.filter(infiles, pat)
            fffil = []  # Second filter
            for f in filtfil:
                if 'av' in f.split('.'):
                    # Is already computed for this station
                    # So let's remove all of them from the initial list
                    for ff in filtfil:
                        infiles.remove(ff)
                    filtfil.clear()
                    fffil.clear()
                    self.logger.debug('Skipping already averaged dv...%s' % f)
                    break
                if method.lower() == 'autocomponents':
                    # Remove those from combined channels
                    components = f.split('.')[-2].split('-')
                    if components[0] != components[1]:
                        continue
                fffil.append(f)
            dvs = []
            for f in fffil:
                try:
                    dvs.append(read_dv(f))
                except ValueError:
                    raise ValueError(f)
                # Remove so they are not processed again
                infiles.remove(f)
            if not len(dvs):
                continue
            elif len(dvs) == 1:
                self.logger.warn(
                    'Only one component found for station %s.%s... Skipping.'
                    % (dvs[0].stats.network, dvs[0].stats.station))
                continue
            dv_av = average_components(dvs)
            outf = os.path.join(
                self.outdir, 'DV-%s.%s.av' % (
                    dv_av.stats.network, dv_av.stats.station))
            dv_av.save(outf)
            if self.options['dv']['plot_vel_change']:
                # plot if desired
                fname = '%s_%s_av' % (
                    dv_av.stats.network, dv_av.stats.station)
                savedir = os.path.join(
                    self.options['proj_dir'], self.options['fig_subdir'])
                dv_av.plot(
                    save_dir=savedir, figure_file_name=fname,
                    normalize_simmat=True, sim_mat_Clim=[-1, 1])


def make_time_list(
    start_date: str, end_date: str, date_inc: int, win_len: int) -> Tuple[
        np.ndarray, np.ndarray]:
    """
    Returns numpy array of starttimes and endtimes, for which each data point
    of velocity change should be computed.

    :param start_date: Date (including time) to start monitoring
    :type start_date: str
    :param end_date: Last date (including time) of monitoring
    :type end_date: str
    :param date_inc: Increment in seconds between each datapoint
    :type date_inc: int
    :param win_len: Window Length, for which each data point is computed.
    :type win_len: int
    :return: A numpy array holding the starttimes and endtimes of the computing
        time windows in seconds (as UTC timestamps)
    :rtype: Tuple[ np.ndarray, np.ndarray]

    ..note::

        see `obspy's documentation
        <https://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html>`
        for compatible input strings.
    """
    if date_inc <= 0 or win_len <= 0:
        raise ValueError(
            'Negative increments or window length are not allowed.')
    start = UTCDateTime(start_date).timestamp
    end = UTCDateTime(end_date).timestamp
    if start >= end:
        raise ValueError('Start Date should be earlier than end date.')
    inc = date_inc
    starttimes = np.arange(start, end, inc)
    endtimes = starttimes + win_len
    return starttimes, endtimes


def corr_find_filter(indir: str, net: dict, **kwargs) -> Tuple[
        List[str], List[str], List[str]]:
    """
    1. Finds the files of available correlations.
    2. Out of those files, it will extract the ones that are requested.

    :param indir: Directory that the correlations are saved in.
    :type indir: str
    :param net: Network dictionary from *yaml* file.
    :type net: dict
    :return: list of network combination codes, list of station combination
        codes, and list of the file paths to their corresponding
        correlation files.
    :rtype: Tuple[List[str], List[str], List[str]]
    """
    netlist = []
    statlist = []
    infiles = []
    for f in glob(os.path.join(indir, '*-*.*-*.h5')):
        f = os.path.basename(f)
        split = f.split('.')

        # Find the files that should actually be processed
        nsplit = split[0].split('-')
        ssplit = split[1].split('-')
        if isinstance(net['network'], str) and not fnmatch.filter(
                nsplit, net['network']) == nsplit:
            continue
        elif isinstance(net['network'], list) and (
                nsplit[0] and nsplit[1]) not in net['network']:
            continue
        if isinstance(net['station'], str) and not fnmatch.filter(
                ssplit, net['station']) == ssplit:
            continue
        elif isinstance(net['station'], list) and (
                ssplit[0] and ssplit[1]) not in net['station']:
            continue

        netlist.append(split[0])
        statlist.append(split[1])
        infiles.append(os.path.join(indir, f))
    return netlist, statlist, infiles


def average_components(dvs: List[DV]) -> DV:
    """
    Averages the Similariy matrix of the three DV objects. Based on those,
    it computes a new dv value and a new correlation value.

    :param dvs: List of dvs from the different components to compute an
            average from. Note that it is possible to use almost anything as
            input (also from different stations). However, at the time,
            the function requires the input to be of the same shape

    :type dvs: List[class:`~seismic.monitor.dv.DV`]
    :raises TypeError: for DVs that were computed with different methods
    :return: A single dv with an averaged similarity matrix
    :rtype: DV
    """
    shapes = []
    for dv in dvs:
        if dv.method != dvs[0].method:
            raise TypeError('DV has to be computed with the same method.')
        # adapt shape to maiximum
        shapes.append(dv.sim_mat.shape)
    #     if dv.sim_mat.shape[:-1] != shapes[0][:-1]:
    #         raise ValueError(
    #             'Only the time axis is allowed to have a different shape.' +
    #             'Make sure you use the same number of stretching steps')
    # shape = max(shapes)
    sim_mats = [dv.sim_mat for dv in dvs]
    av_sim_mat = np.nanmean(sim_mats, axis=0)
    # Now we would have to recompute the dv value and corr value
    corr = np.nanmax(av_sim_mat, axis=1)
    strvec = dvs[0].second_axis
    dt = strvec[np.nanargmax(np.nan_to_num(av_sim_mat), axis=1)]
    stats = deepcopy(dvs[0].stats)
    stats['channel'] = 'av'
    try:
        stats['processing'] = list(
            stats['processing']).append('Averaged Similarity Matrix.')
    except KeyError:
        stats['processing'] = [('Averaged Similarity Matrix.')]
    dvout = DV(
        corr, dt, dvs[0].value_type, av_sim_mat, strvec, dvs[0].method, stats)
    return dvout
