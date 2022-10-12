'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 3rd June 2021 04:15:57 pm
Last Modified: Wednesday, 12th October 2022 10:31:25 am
'''
from copy import deepcopy
import logging
import os
from typing import Generator, List, Tuple
import warnings
import yaml
import fnmatch
from glob import glob

from mpi4py import MPI
import numpy as np
from obspy import UTCDateTime
from tqdm import tqdm

from seismic.db.corr_hdf5 import CorrelationDataBase
from seismic.monitor.dv import DV, read_dv
from seismic.monitor.wfc import WFC
from seismic.utils.miic_utils import log_lvl
import seismic.monitor.post_corr_process as pcp


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
        # replace colon in tstr
        tstr = tstr.replace(':', '-')
        self.logger = logging.getLogger(
            "seismic.monitor.Monitor%s" % str(self.rank).zfill(3))
        self.logger.setLevel(loglvl)

        # also catch the warnings
        logging.captureWarnings(True)
        warnlog = logging.getLogger('py.warnings')
        fh = logging.FileHandler(os.path.join(
            logdir, f'monitor{tstr}rank{str(self.rank).zfill(3)}'))
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
        self.netlist, self.statlist, self.infiles = \
            self._find_available_corrs()

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
        self.logger.info(
            'Found correlation data for the following station '
            + 'and network combinations %s' % str(
                ['{n}.{s}'.format(n=n, s=s) for n, s in zip(
                    netlist, statlist)]))
        return netlist, statlist, infiles

    def compute_velocity_change(
        self, corr_file: str, tag: str, network: str, station: str,
            channel: str, ref_trcs: np.ndarray = None):
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
        :param ref_trcs: Feed in on or several custom reference traces.
            If None the program will determine a reference trace from
            the chosen method in the config. Defaults to None
        :type ref_trcs: np.ndarray, optional
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
        if 'preprocessing' in self.options['dv']:
            for func in self.options['dv']['preprocessing']:
                f = cb.__getattribute__(func['function'])
                cb = f(**func['args'])

        # Now, we make a copy of the cm to be trimmed
        if self.options['dv']['tw_len'] is None:
            trim0 = cb.stats.start_lag
            trim1 = cb.stats.end_lag
            cbt = cb
        else:
            trim0 = -(
                self.options['dv']['tw_start']+self.options['dv']['tw_len'])
            trim1 = (
                self.options['dv']['tw_start']+self.options['dv']['tw_len'])
            cbt = cb.copy().trim(trim0, trim1)

        if cbt.data.shape[1] <= 20:
            raise ValueError('CorrBulk extremely short.')

        if ref_trcs is None:
            tr = cbt.extract_multi_trace(**self.options['dv']['dt_ref'])
        else:
            tr = ref_trcs
            if max(tr.shape) != cbt.data.shape[1]:
                tr, _ = pcp.corr_mat_trim(tr, deepcopy(cb.stats), trim0, trim1)

        # Compute time window
        tw = [np.arange(
            self.options['dv']['tw_start']*cbt.stats['sampling_rate'],
            trim1*cbt.stats['sampling_rate'], 1)]
        dv = cbt.stretch(
            ref_trc=tr, return_sim_mat=True,
            stretch_steps=self.options['dv']['stretch_steps'],
            stretch_range=self.options['dv']['stretch_range'],
            tw=tw, sides=self.options['dv']['sides'])
        ccb = cb.correct_stretch(dv, self.options['dv']['sides'] == 'single')

        ccb.trim(trim0, trim1)

        # extract the final reference trace (mean excluding very different
        # traces)
        if ref_trcs is None:
            tr = ccb.extract_multi_trace(**self.options['dv']['dt_ref'])

        # obtain an improved time shift measurement
        dv = cbt.stretch(
            ref_trc=tr, return_sim_mat=True,
            stretch_steps=self.options['dv']['stretch_steps'],
            stretch_range=self.options['dv']['stretch_range'],
            tw=tw, sides=self.options['dv']['sides'])

        # Postprocessing on the dv object
        if 'postprocessing' in self.options['dv']:
            for func in self.options['dv']['postprocessing']:
                f = dv.__getattribute__(func['function'])
                dv = f(**func['args'])

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
            except KeyError:
                self.logger.exception(
                    f'No correlation data found for {net}.{stat}.{cha} with '
                    + f'tag {tag} in file {corr_file}.'
                )
            except Exception as e:
                self.logger.exception(f'{e} for file {corr_file}.')

    def compute_components_average(self, method: str = 'AutoComponents'):
        """
        Averages the Similarity matrix of different velocity changes in
        the whole dv folder. Based upon those values, new dvs and correlations
        are computed.

        :param method: Which components should be averaged? Can be
            'StationWide' if all component-combinations for one station should
            be averaged, 'AutoComponents' if only the dv results of
            autocorrelations are to be averaged, 'CrossComponents' if you wish
            to average dvs from the same station but only intercomponent
            corrrelations, or 'CrossStations' if cross-station correlations
            should be averaged (same station combination and all component
            combinations).
            Defaults to 'AutoComponents'
        :type method: str, optional
        :raises ValueError: For Unknown combination methods.
        """
        av_methods = (
            'autocomponents', 'crosscomponents', 'stationwide',
            'crossstations', 'betweenstations', 'betweencomponents')
        if method.lower() not in av_methods:
            raise ValueError('Averaging method not in %s.' % str(av_methods))
        infiles = glob(os.path.join(self.outdir, '*.npz'))
        if method.lower() == 'autocomponents':
            ch = 'av-auto'
        elif method.lower() in ('betweencomponents', 'crosscomponents'):
            ch = 'av-xc'
        else:
            ch = 'av'
        while len(infiles):
            # Find files belonging to same station
            pat = '.'.join(infiles[0].split('.')[:-2]) + '*'
            filtfil = fnmatch.filter(infiles, pat)
            fffil = []  # Second filter
            for f in filtfil:
                if ch in f.split('.'):
                    # Is already computed for this station
                    # So let's remove all of them from the initial list
                    for ff in filtfil:
                        try:
                            infiles.remove(ff)
                        except ValueError:
                            # Has already been removed by one of the other
                            # check
                            pass
                    filtfil.clear()
                    fffil.clear()
                    self.logger.debug('Skipping already averaged dv...%s' % f)
                    break
                elif 'av' in f:
                    # computed by another averaging method
                    infiles.remove(f)
                    continue
                components = f.split('.')[-2].split('-')
                stations = f.split('.')[-3].split('-')
                if method.lower() == 'autocomponents':
                    # Remove those from combined channels
                    if components[0] != components[1] \
                            or stations[0] != stations[1]:
                        infiles.remove(f)
                        continue
                elif method.lower() in (
                        'betweencomponents', 'crosscomponents'):
                    # Remove those from equal channels
                    components = f.split('.')[-2].split('-')
                    if components[0] == components[1] \
                            or stations[0] != stations[1]:
                        infiles.remove(f)
                        continue
                elif method.lower() == 'stationwide':
                    if stations[0] != stations[1]:
                        infiles.remove(f)
                        continue
                elif method.lower() in ('betweenstations', 'crossstations'):
                    if stations[0] == stations[1]:
                        infiles.remove(f)
                        continue
                fffil.append(f)
            dvs = []
            for f in fffil:
                try:
                    dvs.append(read_dv(f))
                except Exception:
                    self.logger.exception(
                        'An unexpected error has ocurred '
                        + 'while reading file %s.' % f)
                # Remove so they are not processed again
                infiles.remove(f)
            if not len(dvs):
                continue
            elif len(dvs) == 1:
                self.logger.warn(
                    'Only component %s found for station %s.%s... Skipping.'
                    % (dvs[0].stats.channel, dvs[0].stats.network,
                        dvs[0].stats.station))
                continue
            dv_av = average_components(dvs)
            outf = os.path.join(
                self.outdir, 'DV-%s.%s.%s' % (
                    dv_av.stats.network, dv_av.stats.station, ch))
            dv_av.save(outf)
            if self.options['dv']['plot_vel_change']:
                # plot if desired
                fname = '%s_%s_%s' % (
                    dv_av.stats.network, dv_av.stats.station, ch)
                savedir = os.path.join(
                    self.options['proj_dir'], self.options['fig_subdir'])
                dv_av.plot(
                    save_dir=savedir, figure_file_name=fname,
                    normalize_simmat=True, sim_mat_Clim=[-1, 1])

    def compute_waveform_coherency_bulk(self):
        """
        Compute the WFC for all specified (params file) correlations using MPI.
        The parameters for the correlation defined in the *yaml* file will be
        used.

        This function will just call
        :meth:`~seismic.monitor.monitor.Monitor.compute_waveform_coherency`
        several times.
        Subsequently, the average of the different component combinations will
        be computed.
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
        wfcl = []
        for ii in tqdm(ind):
            corr_file, net, stat, cha = plist[ii]
            try:
                wfcl.append(self.compute_waveform_coherency(
                    corr_file, tag, net, stat, cha))
            except Exception as e:
                self.logger.exception(e)

        outdir = os.path.join(
            self.options['proj_dir'], self.options['wfc']['subdir'])
        # Compute averages and everything
        wfclu = self.comm.allgather(wfcl)
        # concatenate
        wfcl = [j for i in wfclu for j in i]
        del wfclu

        wfc_avl = []
        while len(wfcl):
            net = wfcl[0].stats.network
            stat = wfcl[0].stats.station
            # find identical components
            wfc_avl = [wfc for wfc in wfcl if all(
                [wfc.stats.network == net, wfc.stats.station == stat])]
            # remove original from list
            list(wfcl.pop(ii) for ii in (wfcl.index(wfc) for wfc in wfc_avl))
            wfc = average_components_wfc(wfc_avl)

            # Write files
            outf = os.path.join(outdir, 'WFC-%s.%s.%s.f%a-%a.tw%a-%a' % (
                wfc.stats.network, wfc.stats.station, wfc.stats.channel,
                wfc.stats.freq_min, wfc.stats.freq_max, wfc.stats.tw_start,
                wfc.stats.tw_len))
            wfc.save(outf)

    def compute_waveform_coherency(
        self, corr_file: str, tag: str, network: str, station: str,
            channel: str) -> WFC:
        """
        Computes the waveform coherency corresponding to one correlation (i.e.,
        one single file).

        The waveform coherency can be used as a measure of stability of
        a certain correlation. See Steinmann, et. al. (2021) for details.

        :param corr_file: File to compute the wfc from
        :type corr_file: str
        :param tag: Tag inside of hdf5 file to retrieve corrs from.
        :type tag: str
        :param network: Network combination code.
        :type network: str
        :param station: Station combination code
        :type station: str
        :param channel: Channel combination code.
        :type channel: str
        :raises ValueError: For short correlations
        :return: An object holding the waveform coherency
        :rtype: :class:`~seismic.monitor.wfc.WFC`

        .. seealso:: To compute wfc for several correlations use:
            :meth:`~seismic.monitor.monitor.Monitor.compute.waveform_coherency\
            _bulk`.
        """
        self.logger.info('Computing wfc for file: %s and channel: %s' % (
            corr_file, channel))
        with CorrelationDataBase(corr_file, mode='r') as cdb:
            # get the corrstream containing all the corrdata for this combi
            cst = cdb.get_data(network, station, channel, tag)
        cb = cst.create_corr_bulk(inplace=True)
        outdir = os.path.join(
            self.options['proj_dir'], self.options['wfc']['subdir'])
        if self.rank == 0:
            os.makedirs(outdir, exist_ok=True)

        # for possible rest bits
        del cst
        # Do the actual processing:
        cb.normalize(normtype='absmax')
        # That is were the stacking is happening
        starttimes, endtimes = make_time_list(
            self.options['wfc']['start_date'], self.options['wfc']['end_date'],
            self.options['wfc']['date_inc'], self.options['wfc']['win_len'])
        self.logger.debug('Timelist created.')
        cb.resample(starttimes, endtimes)
        cb.filter(
            (self.options['wfc']['freq_min'], self.options['wfc']['freq_max']))

        # Preprocessing on the correlation bulk
        if 'preprocessing' in self.options['wfc']:
            for func in self.options['wfc']['preprocessing']:
                f = cb.__getattribute__(func['function'])
                cb = f(**func['args'])

        # Now, we make a copy of the cm to be trimmed
        if self.options['wfc']['tw_len'] is not None:
            cbt = cb.copy().trim(
                -(self.options['wfc']['tw_start']
                    + self.options['wfc']['tw_len']),
                (self.options['wfc']['tw_start']
                    + self.options['wfc']['tw_len']))
        else:
            cbt = cb
            self.options['dv']['tw_len'] = cb.stats.end_lag \
                - self.options['wfc']['tw_start']

        self.logger.debug(
            'Preprocessing finished.\nExtracting reference trace...')

        if cbt.data.shape[1] <= 20:
            raise ValueError('CorrBulk extremely short.')

        tr = cbt.extract_multi_trace(**self.options['wfc']['dt_ref'])
        self.logger.debug('Reference trace created.\nComputing WFC...')

        # Compute time window
        tw = [np.arange(
            self.options['wfc']['tw_start']*cbt.stats['sampling_rate'],
            (self.options['wfc']['tw_start']+self.options[
                'wfc']['tw_len'])*cbt.stats['sampling_rate'], 1)]
        wfc = cbt.wfc(
            tr, tw, 'both', self.options['wfc']['tw_start'],
            self.options['wfc']['tw_len'], self.options['wfc']['freq_min'],
            self.options['wfc']['freq_max'])
        self.logger.debug('WFC computed.\nAveraging over time axis...')
        # Compute the average over the whole time window
        wfc.compute_average()
        if self.options['wfc']['save_comps']:
            outf = os.path.join(outdir, 'WFC-%s.%s.%s.f%a-%a.tw%a-%a' % (
                network, station, channel, wfc.stats.freq_min,
                wfc.stats.freq_max, wfc.stats.tw_start, wfc.stats.tw_len))
            self.logger.info(f'Writing WFC to {outf}')
            wfc.save(outf)
        return wfc


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


def average_components_mem_save(
        dvs: Generator[DV, None, None], save_scatter: bool = True) -> DV:
    """
    Averages the Similariy matrix of the DV objects. Based on those,
    it computes a new dv value and a new correlation value. Less memory intense
    but slower than :func:`average_components`. Should be used if you run
    out of memory using the former. Uses the sum of squares rule for the
    standard deviation.

    :param dvs: Iterator over dvs from the different components to compute an
        average from. Note that it is possible to use almost anything as
        input as long as the similarity matrices of the dvs have the same shape
    :type dvs: Iterator[class:`~seismic.monitor.dv.DV`]
    :param save_scatter: Saves the scattering of old values and correlation
        coefficients to later provide a statistical measure. Defaults to True.
    :type save_scatter: bool, optional
    :raises TypeError: for DVs that were computed with different methods
    :return: A single dv with an averaged similarity matrix.
    :rtype: DV
    """
    statsl = []
    corrs = []
    stretches = []
    for ii, dv in enumerate(dvs):
        if ii == 0:
            sim_mat_sum = np.zeros_like(dv.sim_mat)
            n_stat = np.zeros_like(dv.corr)

            stats = deepcopy(dv.stats)
            strvec = dv.second_axis
            value_type = dv.value_type
            method = dv.method
        if dv.method != method:
            raise TypeError('DV has to be computed with the same method.')
        if 'av' in dv.stats.channel+dv.stats.network+dv.stats.station:
            warnings.warn('Averaging of averaged dvs not allowed. Skipping dv')
            continue
        if dv.sim_mat.shape != sim_mat_sum.shape or any(
                dv.second_axis != strvec):
            warnings.warn(
                'The shapes of the similarity matrices of the input DVs '
                + 'vary. Make sure to compute the dvs with the same parameters'
                + ' (i.e., start & end dates, date-inc, stretch increment, '
                + 'and stretch steps.\n\nThis dv will be skipped'
            )
            continue
        sim_mat_sum += np.nan_to_num(dv.sim_mat)
        n_stat[~np.isnan(dv.value)] += 1
        if save_scatter:
            corrs.append(dv.corr)
            stretches.append(dv.value)
        statsl.append(dv.stats)
    # Now inf where it was nan before
    av_sim_mat = (sim_mat_sum.T/n_stat).T
    av_sim_mat[np.isinf(av_sim_mat)] = np.nan
    # Now we would have to recompute the dv value and corr value
    iimax = np.nanargmax(np.nan_to_num(av_sim_mat), axis=1)
    corr = np.nanmax(av_sim_mat, axis=1)
    dt = strvec[iimax]
    if save_scatter:
        # use sum of square for std
        corrs = np.array(corrs)
        stretches = np.array(stretches)
    else:
        corrs = None
        stretches = None
    if not all(np.array([st.channel for st in statsl]) == stats.channel):
        stats['channel'] = 'av'
    if not all(np.array([st.station for st in statsl]) == stats.station):
        stats['station'] = 'av'
    if not all(np.array([st.network for st in statsl]) == stats.network):
        stats['network'] = 'av'
    dvout = DV(
        corr, dt, value_type, av_sim_mat, strvec,
        method, stats, stretches=stretches, corrs=corrs,
        n_stat=n_stat)
    return dvout


def average_components(dvs: List[DV], save_scatter: bool = True) -> DV:
    """
    Averages the Similariy matrix of the DV objects. Based on those,
    it computes a new dv value and a new correlation value.

    :param dvs: List of dvs from the different components to compute an
        average from. Note that it is possible to use almost anything as
        input as long as the similarity matrices of the dvs have the same shape
    :type dvs: List[class:`~seismic.monitor.dv.DV`]
    :param save_scatter: Saves the scattering of old values and correlation
        coefficients to later provide a statistical measure. Defaults to True.
    :type save_scatter: bool, optional
    :raises TypeError: for DVs that were computed with different methods
    :return: A single dv with an averaged similarity matrix.
    :rtype: DV

    .. seealso::
        If you should get an `OutOfMemoryError` consider using
        :func:`average_components_mem_save`.
    """
    dv_use = []
    for dv in dvs:
        if dv.method != dvs[0].method:
            raise TypeError('DV has to be computed with the same method.')
        if 'av' in dv.stats.channel+dv.stats.network+dv.stats.station:
            warnings.warn('Averaging of averaged dvs not allowed. Skipping dv')
            continue
        if dv.sim_mat.shape != dvs[0].sim_mat.shape or any(
                dv.second_axis != dvs[0].second_axis):
            warnings.warn(
                'The shapes of the similarity matrices of the input DVs '
                + 'vary. Make sure to compute the dvs with the same parameters'
                + ' (i.e., start & end dates, date-inc, stretch increment, '
                + 'and stretch steps.\n\nThis dv will be skipped'
            )
            continue
        dv_use.append(dv)
    sim_mats = [dv.sim_mat for dv in dv_use]
    av_sim_mat = np.nanmean(sim_mats, axis=0)
    # Now we would have to recompute the dv value and corr value
    iimax = np.nanargmax(np.nan_to_num(av_sim_mat), axis=1)
    corr = np.nanmax(av_sim_mat, axis=1)
    strvec = dv_use[0].second_axis
    dt = strvec[iimax]
    if save_scatter:
        stretches = np.array([dv.value for dv in dv_use])
        corrs = np.array([dv.corr for dv in dv_use])
        # # Number of stations per corr_start
        n_stat = np.ones_like(stretches, dtype=int)
        n_stat[np.where(np.isnan(stretches))] = 0
        n_stat = np.sum(n_stat, axis=0)  # now this has the same shape as std
    else:
        stretches = None
        corrs = None
    stats = deepcopy(dv_use[0].stats)
    if not all(np.array([dv.stats.channel for dv in dv_use]) == stats.channel):
        stats['channel'] = 'av'
    if not all(np.array([dv.stats.station for dv in dv_use]) == stats.station):
        stats['station'] = 'av'
    if not all(np.array([dv.stats.network for dv in dv_use]) == stats.network):
        stats['network'] = 'av'
    dvout = DV(
        corr, dt, dv_use[0].value_type, av_sim_mat, strvec,
        dv_use[0].method, stats, stretches=stretches, corrs=corrs,
        n_stat=n_stat)
    return dvout


def average_dvs_by_coords(
    dvs: List[DV], lat: Tuple[float, float], lon: Tuple[float, float],
        el: Tuple[float, float] = (-1e6, 1e6), return_std: bool = False) -> DV:
    """
    Averages the Similariy matrix of the DV objects if they are inside of the
    queried coordionates. Based on those,
    it computes a new dv value and a new correlation value.

    :param dvs: List of dvs from the different components to compute an
        average from. Note that it is possible to use almost anything as
        input as long as the similarity matrices of the dvs have the same shape
    :type dvs: List[class:`~seismic.monitor.dv.DV`]
    :param lat: Minimum and maximum latitude of the stations(s). Note that
        for cross-correlations both stations must be inside of the window.
    :type lat: Tuple[float, float]
    :param lon: Minimum and maximum longitude of the stations(s). Note that
        for cross-correlations both stations must be inside of the window.
    :type lon: Tuple[float, float]
    :param el: Minimum and maximum elevation of the stations(s) in m. Note that
        for cross-correlations both stations must be inside of the window.
        Defaults to (-1e6, 1e6) - i.e., no filter.
    :type el: Tuple[float, float], optional
    :param return_std: Save the standard deviation of the similarity
        matrices into the dv object. Defaults to False.
    :type return_std: bool, optional
    :raises TypeError: for DVs that were computed with different methods
    :return: A single dv with an averaged similarity matrix.
    :rtype: DV
    """
    dv_filt = []
    for dv in dvs:
        s = dv.stats
        if s.channel == 'av':
            warnings.warn('Averaging of averaged dvs not allowed. Skipping dv')
            continue
        if all((
            lat[0] <= s.stla <= lat[1], lat[0] <= s.evla <= lat[1],
            lon[0] <= s.stlo <= lon[1], lon[0] <= s.evlo <= lon[1],
                el[0] <= s.stel <= el[1], el[0] <= s.evel <= el[1])):
            dv_filt.append(dv)
    if not len(dv_filt):
        raise ValueError(
            'No DVs within geographical constraints found.')
    av_dv = average_components(dv_filt, return_std)
    s = av_dv.stats
    # adapt header
    s['network'] = s['station'] = 'geoav'
    s['stel'] = s['evel'] = el
    s['stlo'] = s['evlo'] = lon
    s['stla'] = s['evla'] = lat
    return av_dv


def average_components_wfc(wfcs: List[WFC]) -> WFC:
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
    stats = deepcopy(wfcs[0].stats)
    stats['channel'] = 'av'
    for k in ['tw_start', 'tw_len', 'freq_min', 'freq_max']:
        if any((stats[k] != wfc.stats[k] for wfc in wfcs)):
            raise ValueError('%s is not allowed to differ.' % k)
    meandict = {}
    for k in wfcs[0].keys():
        meandict[k] = np.nanmean([wfc[k] for wfc in wfcs], axis=0)
    wfcout = WFC(meandict, stats)
    return wfcout
