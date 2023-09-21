'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 3rd June 2021 04:15:57 pm
Last Modified: Thursday, 21st September 2023 04:19:04 pm
'''
from copy import deepcopy
import json
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


class Monitor(object):
    def __init__(self, options: dict | str):
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
        try:
            self.save_comps_separately = options['save_comps_separately']
        except KeyError:
            # If it's not specified it's probably an old params.yaml
            self.save_comps_separately = False

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

        # Write the options dictionary to the log file
        if self.rank == 0:
            opt_dump = deepcopy(options)
            # json cannot write the UTCDateTime objects that might be in here
            for step in opt_dump['co']['preProcessing']:
                if 'stream_mask_at_utc' in step['function']:
                    startsstr = [
                        t.format_fissures() for t in step['args']['starts']]
                    step['args']['starts'] = startsstr
                    if 'ends' in step['args']:
                        endsstr = [
                            t.format_fissures() for t in step['args']['ends']]
                        step['args']['ends'] = endsstr
            with open(os.path.join(
                    logdir, 'params%s.txt' % tstr), 'w') as file:
                file.write(json.dumps(opt_dump, indent=1))

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
            lts = cdb.get_corr_options()['corr_args']['lengthToSave']

        tw_start = self.options['dv']['tw_start']

        if self.options['dv']['compute_tt']:
            if not hasattr(cst[0].stats, 'dist'):
                warnings.warn(
                    f'{network}.{station} does not include distance. '
                    'SeisMIC will assume an interstation distance of 0.')
            else:
                # Assume flat earth to include topography
                # Note that dist is in km and elevation information in m
                d = np.sqrt(
                    cst[0].stats.dist**2
                    + ((cst[0].stats.stel-cst[0].stats.evel)/1000)**2)
                tt = round(
                    d/self.options['dv']['rayleigh_wave_velocity'], 0)
                tw_start += tt
                self.logger.info(
                    f'Computed travel time for {network}.{station} is '
                    f'{tt} s. The assumed direct-line distance was {d} km.')
        if lts < tw_start + self.options['dv']['tw_len']:
            reqtw = tw_start + self.options[
                'dv']['tw_len']
            raise ValueError(
                'Requested lapse time window (time window start + time window'
                f' length = {reqtw} contains segments after the Correlation'
                'Function.'
                ' When computing the correlations make sure to set an '
                f'appropriate value for lengthToSave. The value was {lts}.'
                f' The direct-line distance between the stations is {d} km.'
            )

        if 'preprocessing' in self.options['dv']:
            for func in self.options['dv']['preprocessing']:
                # This one goes on the CorrStream
                if func['function'] in ['pop_at_utcs', 'select_time']:
                    f = cst.__getattribute__(func['function'])
                    cst = f(**func['args'])
        cb = cst.create_corr_bulk(
            network=network, station=station, channel=channel, inplace=True)

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
                if func['function'] in ['pop_at_utcs', 'select_time']:
                    continue
                f = cb.__getattribute__(func['function'])
                cb = f(**func['args'])

        # Retain a copy of the stats
        stats_copy = deepcopy(cb.stats)
        if self.options['dv']['tw_len'] is None:
            trim0 = cb.stats.start_lag
            trim1 = cb.stats.end_lag
            cbt = cb
        else:
            trim0 = -(
                tw_start+self.options['dv']['tw_len'])
            trim1 = (
                tw_start+self.options['dv']['tw_len'])
            cbt = cb.copy().trim(trim0, trim1)

        if cbt.data.shape[1] <= 20:
            raise ValueError('CorrBulk extremely short.')

        # 15.11.22
        # Do trimming after stretching of the reference trace to avoid
        # edges having an influence on dv/v estimate
        if ref_trcs is None:
            tr = cb.extract_multi_trace(**self.options['dv']['dt_ref'])
        else:
            tr = ref_trcs

        # Compute time window
        tw = [np.arange(
            tw_start*cbt.stats['sampling_rate'],
            trim1*cbt.stats['sampling_rate'], 1)]
        dv = cbt.stretch(
            ref_trc=tr, return_sim_mat=True,
            stretch_steps=self.options['dv']['stretch_steps'],
            stretch_range=self.options['dv']['stretch_range'],
            tw=tw, sides=self.options['dv']['sides'],
            ref_tr_trim=(trim0, trim1), ref_tr_stats=stats_copy,
            processing=self.options['dv'])
        # if isinstance(self.options['dv']['exclude_above_corr'], float):
        #     thresmask = np.where(
        #         dv.corr > self.options['dv']['exclude_above_corr'])[0]
            
        #     popstarts = dv.stats.corr_start[thresmask]
        #     popends = dv.stats.corr_end[thresmask]
        #     popind = cbt._find_slice_index()
        ccb = cb.correct_stretch(dv)

        # extract the final reference trace (mean excluding very different
        # traces)
        if ref_trcs is None:
            tr = ccb.extract_multi_trace(**self.options['dv']['dt_ref'])

        ccb.trim(trim0, trim1)

        # obtain an improved time shift measurement
        dv = cbt.stretch(
            ref_trc=tr, return_sim_mat=True,
            stretch_steps=self.options['dv']['stretch_steps'],
            stretch_range=self.options['dv']['stretch_range'],
            tw=tw, sides=self.options['dv']['sides'],
            ref_tr_trim=(trim0, trim1), ref_tr_stats=stats_copy,
            processing=self.options['dv'])

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
        pmap = np.arange(len(plist))*self.psize/len(plist)
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

    def compute_waveform_coherence_bulk(self):
        """
        Compute the WFC for all specified (params file) correlations using MPI.
        The parameters for the correlation defined in the *yaml* file will be
        used.

        This function will just call
        :meth:`~seismic.monitor.monitor.Monitor.compute_waveform_coherence`
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
        pmap = np.arange(len(plist))*self.psize/len(plist)
        pmap = pmap.astype(np.int32)
        ind = pmap == self.rank
        ind = np.arange(len(plist), dtype=int)[ind]

        # Assign a task to each rank
        wfcl = []
        for ii in tqdm(ind):
            corr_file, net, stat, cha = plist[ii]
            for wfc in self.compute_waveform_coherence(
                    corr_file, tag, net, stat, cha):
                try:
                    wfcl.append(wfc)
                except Exception as e:
                    self.logger.exception(e)

        outdir = os.path.join(
            self.options['proj_dir'], self.options['wfc']['subdir'])
        # Compute averages and everything
        wfclu = self.comm.allgather(wfcl)
        # concatenate
        wfcl = [j for i in wfclu for j in i]
        del wfclu

        # Find unique averaging groups
        wfc_avl = []
        for wfc in wfcl:
            wfc_avl.append(
                (wfc.stats.network, wfc.stats.station,
                    wfc.wfc_processing['freq_min'],
                    wfc.wfc_processing['freq_max'],
                    wfc.wfc_processing['tw_start'],
                    wfc.wfc_processing['tw_len']))
        wfc_avl = list(set(wfc_avl))
        wfcl_sub = []
        for avl in wfc_avl:
            net, stat, fmin, fmax, tw_start, tw_len = avl
            wfcl_sub.append(
                [wfc for wfc in wfcl if all(
                    [
                        wfc.stats.network == net, wfc.stats.station == stat,
                        wfc.wfc_processing['freq_min'] == fmin,
                        wfc.wfc_processing['freq_max'] == fmax,
                        wfc.wfc_processing['tw_start'] == tw_start,
                        wfc.wfc_processing['tw_len'] == tw_len
                    ])])
        for wfc_avl in wfcl_sub:
            wfc = average_components_wfc(wfc_avl)
            # Write files
            outf = os.path.join(outdir, 'WFC-%s.%s.%s.f%a-%a.tw%a-%a' % (
                wfc.stats.network, wfc.stats.station, wfc.stats.channel,
                wfc.wfc_processing['freq_min'],
                wfc.wfc_processing['freq_max'],
                wfc.wfc_processing['tw_start'],
                wfc.wfc_processing['tw_len']))
            wfc.save(outf)

    def compute_waveform_coherence(
        self, corr_file: str, tag: str, network: str, station: str,
            channel: str) -> WFC:
        """
        Computes the waveform coherence corresponding to one correlation (i.e.,
        one single file).

        The waveform coherence can be used as a measure of stability of
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
        :return: An object holding the waveform coherence
        :rtype: :class:`~seismic.monitor.wfc.WFC`

        .. seealso:: To compute wfc for several correlations use:
            :meth:`~seismic.monitor.monitor.Monitor.compute.waveform_coherence\
            _bulk`.
        """
        self.logger.info('Computing wfc for file: %s and channel: %s' % (
            corr_file, channel))
        with CorrelationDataBase(corr_file, mode='r') as cdb:
            # get the corrstream containing all the corrdata for this combi
            cst = cdb.get_data(network, station, channel, tag)

        if 'preprocessing' in self.options['wfc']:
            for func in self.options['wfc']['preprocessing']:
                # This one goes on the CorrStream
                if func['function'] in ['pop_at_utcs', 'select_time']:
                    f = cst.__getattribute__(func['function'])
                    cst = f(**func['args'])

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

        # Allow lists so the computation does not have to be started x times
        if not isinstance(self.options['wfc']['freq_min'], list):
            self.options['wfc']['freq_min'] = [
                self.options['wfc']['freq_min']]
        if not isinstance(self.options['wfc']['freq_max'], list):
            self.options['wfc']['freq_max'] = [
                self.options['wfc']['freq_max']]
        if len(self.options['wfc']['freq_min']) != len(
                self.options['wfc']['freq_max']):
            raise ValueError(
                'freq_min and freq_max must be of same length.')

        if not isinstance(self.options['wfc']['tw_start'], list):
            self.options['wfc']['tw_start'] = [
                self.options['wfc']['tw_start']]
        if isinstance(self.options['wfc']['tw_len'], list) and len(
                self.options['wfc']['tw_len']) != len(
                self.options['wfc']['tw_start']):
            raise ValueError(
                'tw_start and tw_len must be of same length or tw_len must be'
                ' a single value.')
        elif not isinstance(self.options['wfc']['tw_len'], list):
            self.options['wfc']['tw_len'] = [
                self.options['wfc']['tw_len']] * len(
                self.options['wfc']['tw_start'])

        cb_bac = cb.copy()

        for fmin, fmax in zip(
                self.options['wfc']['freq_min'],
                self.options['wfc']['freq_max']):
            if fmin >= fmax:
                raise ValueError('freq_min must be smaller than freq_max.')
            cb.filter((fmin, fmax))

            # Preprocessing on the correlation bulk
            if 'preprocessing' in self.options['wfc']:
                for func in self.options['wfc']['preprocessing']:
                    if func['function'] in ['pop_at_utcs', 'select_time']:
                        continue
                    f = cb.__getattribute__(func['function'])
                    cb = f(**func['args'])

            self.logger.debug(
                f'Preprocessing finished. Computing wfc for fmin: {fmin} '
                f'and fmax: {fmax} {len(self.options["wfc"]["tw_start"])}'
                f'time windows.')

            # Now, we loop over the time windows
            for tw_start, tw_len in zip(
                self.options['wfc']['tw_start'],
                    self.options['wfc']['tw_len']):
                try:
                    # Now, we make a copy of the cm to be trimmed
                    cbt = cb.copy().trim(
                        -(tw_start + tw_len), tw_start + tw_len)

                    if cbt.data.shape[1] <= 20:
                        raise ValueError('CorrBulk extremely short.')

                    tr = cbt.extract_multi_trace(
                        **self.options['wfc']['dt_ref'])
                    self.logger.debug(
                        'Reference trace created.\nComputing WFC...')

                    # Compute time window
                    tw = [np.arange(
                        tw_start*cbt.stats['sampling_rate'],
                        (tw_start+tw_len)*cbt.stats['sampling_rate'], 1)]
                    wfc = cbt.wfc(
                        tr, tw, 'both', tw_start, tw_len, fmin, fmax)
                    self.logger.debug(
                        'WFC computed.\nAveraging over time axis...')
                    # Compute the average over the whole time window
                    wfc.compute_average()
                    if self.options['wfc']['save_comps']:
                        outf = os.path.join(
                            outdir, 'WFC-%s.%s.%s.f%a-%a.tw%a-%a' % (
                                network, station, channel,
                                fmin, fmax, tw_start, tw_len))
                        self.logger.info(f'Writing WFC to {outf}')
                        wfc.save(outf)
                    yield wfc
                except Exception as e:
                    self.logger.error(
                        'Error computing WFC for fmin: '
                        f'{fmin} and fmax: {fmax}'
                        f' and tw_start: {tw_start} and tw_len: {tw_len}. '
                        f'Error: {e}')
            cb = cb_bac.copy()


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


def corr_find_filter(
    indir: str, net: dict, **kwargs) -> Tuple[
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
        matched = False
        f = os.path.basename(f)
        split = f.split('.')

        # Find the files that should actually be processed
        nsplit = split[0].split('-')
        ssplit = split[1].split('-')
        if isinstance(net['network'], str) and not fnmatch.filter(
                nsplit, net['network']) == nsplit:
            continue
        elif isinstance(net['network'], list):
            # account for the case of wildcards in the network list
            for n in net['network']:
                if fnmatch.filter(nsplit, n) == nsplit:
                    matched = True
                    break
            if not matched:
                continue
        if isinstance(net['station'], str) and not fnmatch.filter(
                ssplit, net['station']) == ssplit:
            continue
        elif isinstance(net['station'], list):
            # account for the case of wildcards in the station list
            for s in net['station']:
                if fnmatch.filter(ssplit, s) == ssplit:
                    matched = True
                    break
            if not matched:
                continue

        netlist.append(split[0])
        statlist.append(split[1])
        infiles.append(os.path.join(indir, f))
    return netlist, statlist, infiles


def correct_dv_shift(
    dv0: DV, dv1: DV, method: str = 'mean',
        n_overlap: int = 0) -> Tuple[DV, DV]:
    """
    Shift dv0 with respect to dv1, so that the same times will have the same
    value.

    .. note::
        in-place operation.

    :param dv0: DV object to be shifted
    :type dv0: DV
    :param dv1: DV object to compare to
    :type dv1: DV
    :param method: either mean or median, defaults to 'mean'. If mean is chosen
        the points will be weighted by the correlation coefficient.
    :type method: str, optional
    :param n_overlap: Number of points to compare to beyond the
        already overlapping point. Makes sense if, e.g., one station replaces
        the other, defaults to 0.
    :type n_overlap: int, optional
    :raises ValueError: If the two dvs only have non-nan value more than
        n_overlap apart.
    :return: The two dv object. Only the first one is modified. Modification
        also happens in-place.
    :rtype: Tuple[DV, DV]
    """
    # Times when both measured
    both_avail = np.all(np.vstack((dv0.avail, dv1.avail)), axis=0)
    if np.any(both_avail):
        # Unless one of them become less than 0 or more than length
        start = np.where(both_avail)[0].min() - n_overlap
        stop = np.where(both_avail)[0].max() + n_overlap
        ii = jj = 0
    elif (
        np.array(dv0.stats.starttime)[dv0.avail][0]
            > np.array(dv1.stats.starttime)[dv1.avail][-1]):
        # dv0 comes after dv1:
        ii = np.where(
            np.array(dv0.stats.starttime) == np.array(
                dv0.stats.starttime)[dv0.avail][0])[0][0]
        stop = ii + n_overlap
        jj = np.where(
            np.array(dv1.stats.starttime) == np.array(
                dv1.stats.starttime)[dv1.avail][-1])[0][0]
        start = jj - n_overlap
    else:
        # dv1 comes after dv0:
        ii = np.where(
            np.array(dv1.stats.starttime) == np.array(
                dv1.stats.starttime)[dv1.avail][0])[0][0]
        stop = ii + n_overlap
        jj = np.where(
            np.array(dv0.stats.starttime) == np.array(
                dv0.stats.starttime)[dv0.avail][-1])[0][0]
        start = jj - n_overlap
    if ii-jj > n_overlap:
        raise ValueError(
            'The gap in active times of the two DVs is larger than '
            'n_overlap, i.e., '
            f'{n_overlap*(dv0.stats.starttime[1]-dv0.stats.starttime[0])/3600}'
            'hours. The result would not make sense.')
    if stop >= len(dv0.value):
        stop = len(dv0.value) - 1
    if start < 0:
        start = 0
    dv0c = dv0.corr[start:stop]
    dv0v = dv0.value[start:stop]
    dv1c = dv1.corr[start:stop]
    dv1v = dv1.value[start:stop]
    if method == 'mean':
        shift = np.nanmean(dv0v*dv0c)/np.nanmean(dv0c)\
            - np.nanmean(dv1v*dv1c)/np.nanmean(dv1c)
    elif method == 'median':
        shift = np.nanmedian(
            dv0.value[start:stop])-np.nanmedian(dv1.value[start:stop])
    else:
        raise ValueError('Method has to be either mean or median')
    roll = int(round(-shift/(dv0.second_axis[1]-dv0.second_axis[0])))
    dv0.sim_mat = np.roll(dv0.sim_mat, (roll, 0))
    dv0.value = dv0.second_axis[
        np.nanargmax(np.nan_to_num(dv0.sim_mat), axis=1)]
    return dv0, dv1


def correct_time_shift_several(dvs: List[DV], method: str, n_overlap: int):
    """
    Suppose you have a number of dv/v time series from the same location,
    but station codes changed several times. To stitch these together,
    this code will iteratively shift each progressive time-series to the
    shift of the end of its precessor.

    .. note::
        The shifting correction happens in-place.

    :param dvs: List of dvs to be shifted
    :type dvs: List[DV]
    :param method: method to be used to measure the shift. Can be 'mean' or
        'median'. If 'mean' is chosen, the points will be weighted by their
        respective correlation coefficient.
    :type method: str
    :param n_overlap: amount of points beyond the overlap to use to compute
        the shift.
    :type n_overlap: int
    """
    # If three different time-series are provided, find the one that
    # spans the middle time
    avl = np.array([
        np.mean(np.array([
            t.timestamp for t in dv.stats.corr_start])[
                dv.avail]) for dv in dvs])
    avl_u = np.sort(np.unique(avl))
    # Loop over each unique start and shift and correct them iteratively
    for k, avstart in enumerate(avl_u):
        if k+1 == len(avl_u):
            # everything is shifted
            break
        ii_ref = np.concatenate(np.where(avl == avstart))[0]
        ii_corr = np.concatenate(np.where(avl == avl_u[k+1]))
        dv_r = dvs[ii_ref]
        dv_corr = [dvs[ii] for ii in ii_corr]
        for dv in dv_corr:
            try:
                # The correction should happen in-place
                correct_dv_shift(
                    dv, dv_r, method=method,
                    n_overlap=n_overlap)
            except ValueError as e:
                warnings.warn(
                    f'{e} for {dv.stats.id} and reference dv '
                    f'{dv_r.stats.id}.')


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
            dv_processing = dv.dv_processing
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
    dt[np.isnan(corr)] = np.nan
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
        n_stat=n_stat, dv_processing=dv_processing)
    return dvout


def average_components(
    dvs: List[DV], save_scatter: bool = True, correct_shift: bool = False,
    correct_shift_method: str = 'mean',
        correct_shift_overlap: int = 0) -> DV:
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
    :param correct_shift: Shift the dvs with respect to each other. This is
        relevant if not all stations are active all the time and the references
        are not describing the same state. This only works if stations were
        active almost simultaneously at least for a bit
        (i.e., corr-start[n+n_overlap]-corr-start[n])
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
    # Correct shift
    if correct_shift:
        correct_time_shift_several(
            dv_use, method=correct_shift_method,
            n_overlap=correct_shift_overlap)
    av_sim_mat = np.nanmean([dv.sim_mat for dv in dv_use], axis=0)
    # Now we would have to recompute the dv value and corr value
    iimax = np.nanargmax(np.nan_to_num(av_sim_mat), axis=1)
    corr = np.nanmax(av_sim_mat, axis=1)
    strvec = dv_use[0].second_axis
    dt = strvec[iimax]
    dt[np.isnan(corr)] = np.nan
    if save_scatter:
        stretches = np.array([dv.value for dv in dv_use])
        corrs = np.array([dv.corr for dv in dv_use])
    else:
        stretches = None
        corrs = None
    # Number of stations per corr_start
    n_stat = np.array(
        [dv.avail*np.ones_like(dv.corr, dtype=int) for dv in dv_use])
    n_stat = np.sum(n_stat, axis=0)
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
        n_stat=n_stat, dv_processing=dv_use[0].dv_processing)
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
    wfcp = wfcs[0].wfc_processing
    stats['channel'] = 'av'
    for k in ['tw_start', 'tw_len', 'freq_min', 'freq_max']:
        if any((wfcp[k] != wfc.wfc_processing[k] for wfc in wfcs)):
            raise ValueError('%s is not allowed to differ.' % k)
    meandict = {}
    for k in wfcs[0].keys():
        meandict[k] = np.nanmean([wfc[k] for wfc in wfcs], axis=0)
    wfcout = WFC(meandict, stats, wfcp)
    return wfcout
