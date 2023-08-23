'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 29th March 2021 07:58:18 am
Last Modified: Wednesday, 23rd August 2023 02:23:06 pm
'''
from copy import deepcopy
from typing import Iterator, List, Tuple, Optional
from warnings import warn
import os
import logging
import json
import warnings
import yaml

from mpi4py import MPI
import numpy as np
from obspy import Stream, UTCDateTime, Inventory, Trace
from tqdm import tqdm

from seismic.correlate.stream import CorrTrace, CorrStream
from seismic.correlate import preprocessing_td as pptd
from seismic.correlate import preprocessing_stream as ppst
from seismic.db.corr_hdf5 import CorrelationDataBase
from seismic.trace_data.waveform import Store_Client
from seismic.utils.fetch_func_from_str import func_from_str
from seismic.utils import miic_utils as mu


class Correlator(object):
    """
    Object to manage the actual Correlation (i.e., Green's function retrieval)
    for the database.
    """
    def __init__(self, store_client: Store_Client, options: dict or str):
        """
        Initiates the Correlator object. When executing
        :func:`~seismic.correlate.correlate.Correlator.pxcorr()`, it will
        actually compute the correlations and save them in an hdf5 file that
        can be handled using
        :class:`~seismic.db.corr_hdf5.CorrelationDataBase`.
        Data has to be preprocessed before calling this (i.e., the data already
        has to be given in an ASDF format). Consult
        :class:`~seismic.trace_data.preprocess.Preprocessor` for information on
        how to proceed with this.

        :param options: Dictionary containing all options for the correlation.
            Can also be a path to a yaml file containing all the keys required
            in options.
        :type options: dict or str
        """
        if isinstance(options, str):
            with open(options) as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        # init MPI
        self.comm = MPI.COMM_WORLD
        self.psize = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        # directories
        self.proj_dir = options['proj_dir']
        self.corr_dir = os.path.join(self.proj_dir, options['co']['subdir'])
        logdir = os.path.join(self.proj_dir, options['log_subdir'])
        if self.rank == 0:
            os.makedirs(self.corr_dir, exist_ok=True)
            os.makedirs(logdir, exist_ok=True)

        # Logging - rank dependent
        if self.rank == 0:
            tstr = UTCDateTime.now().strftime('%Y-%m-%d-%H:%M')
        else:
            tstr = None
        tstr = self.comm.bcast(tstr, root=0)

        rankstr = str(self.rank).zfill(3)

        loglvl = mu.log_lvl[options['log_level'].upper()]
        self.logger = logging.getLogger("seismic.Correlator%s" % rankstr)
        self.logger.setLevel(loglvl)
        logging.captureWarnings(True)
        warnlog = logging.getLogger('py.warnings')
        fh = logging.FileHandler(os.path.join(logdir, 'correlate%srank%s' % (
            tstr, rankstr)))
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

        self.options = options['co']

        # requested combis?
        if 'xcombinations' in self.options:
            self.rcombis = self.options['xcombinations']
            if self.rcombis == 'None':
                # cumbersome, but someone used it wrong so let's hardcode
                self.rcombis = None
        else:
            self.rcombis = None

        # find the available data
        network = options['net']['network']
        station = options['net']['station']

        # Store_Client
        self.store_client = store_client

        if isinstance(station, list) and len(station) == 1:
            station = station[0]
        if isinstance(network, list) and len(network) == 1:
            network = network[0]

        if (
            network == '*'
                and isinstance(station, str) and '*' not in station):
            raise ValueError(
                'Stations has to be either: \n'
                + '1. A list of the same length as the list of networks.\n'
                + '2. \'*\' That is, a wildcard (string).\n'
                + '3. A list and network is a string describing one '
                + 'station code.')
        elif isinstance(station, str) and isinstance(network, str):
            station = [[network, station]]
        elif station == '*' and isinstance(network, list):
            station = []
            for net in network:
                station.extend(store_client.get_available_stations(net))
        elif isinstance(network, list) and isinstance(station, list):
            if len(network) != len(station):
                raise ValueError(
                    'Stations has to be either: \n'
                    + '1. A list of the same length as the list of networks.\n'
                    + '2. \'*\' That is, a wildcard (string).\n'
                    + '3. A list and network is a string describing one '
                    + 'station code.')
            station = list([n, s] for n, s in zip(network, station))
        elif isinstance(station, list) and isinstance(network, str):
            for ii, stat in enumerate(station):
                station[ii] = [network, stat]
        else:
            raise ValueError(
                'Stations has to be either: \n'
                + '1. A list of the same length as the list of networks.\n'
                + '2. \'*\' That is, a wildcard (string).\n'
                + '3. A list and network is a string describing one '
                + 'station code.')
        if self.rank == 0:
            self.avail_raw_data = []
            for net, stat in station:
                self.avail_raw_data.extend(
                    self.store_client._translate_wildcards(net, stat))
            # make sure this only contains unique combinations
            # with several cores it added entries several times, don't know
            # why?
            self.avail_raw_data = np.unique(
                self.avail_raw_data, axis=0).tolist()
        else:
            self.avail_raw_data = None
        self.avail_raw_data = self.comm.bcast(
            self.avail_raw_data)
        self.station = np.unique(np.array([
            [d[0], d[1]] for d in self.avail_raw_data]), axis=0).tolist()
        self.logger.debug(
            'Fetching data from the following stations:\n%a' % [
                f'{n}.{s}' for n, s in self.station])

        self.sampling_rate = self.options['sampling_rate']

    def find_interstat_dist(self, dis: float):
        """
        Find stations in database with interstation distance smaller than
        dis.

        If no station inventories are available, they will be downloaded.

        :param dis: Find all Stations with distance less than `dis` [in m]
        :type dis: float

        .. note:: only the subset of the in ``params.yaml`` defined
            networks and stations will be queried.
        """
        if not self.options['combination_method'] == 'betweenStations':
            raise ValueError(
                'This function is only available if combination method '
                + 'is set to "betweenStations".')
        # list of requested combinations
        self.rcombis = []
        # Update the store clients invetory
        self.store_client.read_inventory()
        for ii, (n0, s0) in enumerate(self.station):
            inv0 = self.store_client.select_inventory_or_load_remote(n0, s0)
            for n1, s1 in self.station[ii:]:
                inv1 = self.store_client.select_inventory_or_load_remote(
                    n1, s1)
                if mu.filter_stat_dist(inv0, inv1, dis):
                    self.rcombis.append('%s-%s.%s-%s' % (n0, n1, s0, s1))

    def find_existing_times(self, tag: str, channel: str = '*') -> dict:
        """
        Returns the already existing starttimes in form of a dictionary (see
        below)

        :param tag: The tag that the waveforms are saved under
        :type tag: str
        :param channel: Channel Combination Code (e.g., CH0-CH1),
            wildcards accepted. Defaults to '*'
        :type channel: str, optional
        :return: Dictionary that is structured as in the example below
        :rtype: dict

        Examples
        --------
        >>> out_dict = my_correlator.find_existing_times('mytag', 'BHZ-BHH')
        >>> print(out_dict)
        {'NET0.STAT0': {
            'NET1.STAT1': {'BHZ-BHH': [%list of starttimes] ,
            'NET2.STAT2': {'BHZ-BHH':[%list of starttimes]}}}
        """
        netlist, statlist = list(zip(*self.station))
        netcombs, statcombs = compute_network_station_combinations(
            netlist, statlist, method=self.options['combination_method'],
            combis=self.rcombis)
        ex_dict = {}
        for nc, sc in zip(netcombs, statcombs):
            outf = os.path.join(
                self.corr_dir, '%s.%s.h5' % (nc, sc))
            if not os.path.isfile(outf):
                continue
            with CorrelationDataBase(
                    outf, corr_options=self.options, mode='r') as cdb:
                d = cdb.get_available_starttimes(nc, sc, tag, channel)
            s0, s1 = sc.split('-')
            n0, n1 = nc.split('-')
            ex_dict.setdefault('%s.%s' % (n0, s0), {})
            ex_dict['%s.%s' % (n0, s0)]['%s.%s' % (n1, s1)] = d
        return ex_dict

    def pxcorr(self):
        """
        Start the correlation with the parameters that were defined when
        initiating the object.
        """
        cst = CorrStream()
        if self.rank == 0:
            self.logger.debug('Reading Inventory files.')
        # Fetch station coordinates
        if self.rank == 0:
            try:
                inv = self.store_client.read_inventory()
            except Exception as e:
                if self.options['remove_response']:
                    raise FileNotFoundError(
                        'No response information could be found.'
                        + 'If you set remove_response to True, you will need'
                        + 'a station inventory.')
                logging.warning(e)
                warnings.warn(
                    'No Station Inventory found. Proceeding without.')
                inv = None
        else:
            inv = None
        inv = self.comm.bcast(inv, root=0)

        for st, write_flag in self._generate_data():
            cst.extend(self._pxcorr_inner(st, inv))
            if write_flag:
                self.logger.debug('Writing Correlations to file.')
                # Here, we can recombine the correlations for the read_len
                # size (i.e., stack)
                # Write correlations to HDF5
                if self.options['subdivision']['recombine_subdivision'] and \
                        cst.count():
                    stack = cst.stack(regard_location=False)
                    tag = 'stack_%s' % str(self.options['read_len'])
                    self._write(stack, tag)
                if self.options['subdivision']['delete_subdivision']:
                    cst.clear()
                elif cst.count():
                    self._write(cst, tag='subdivision')
                    cst.clear()

        # write the remaining data
        if self.options['subdivision']['recombine_subdivision'] and \
                cst.count():
            self._write(cst.stack(regard_location=False), tag)
        if not self.options['subdivision']['delete_subdivision'] and \
                cst.count():
            self._write(cst, tag='subdivision')

    def _pxcorr_inner(self, st: Stream, inv: Inventory) -> CorrStream:
        """
        Inner loop of pxcorr. Don't call this function!
        """

        # We start out by moving the stream into a matrix
        self.logger.debug(
            'Converting Stream to Matrix')
        # put all the data into a single stream
        starttime = []
        npts = []
        for tr in st:
            starttime.append(tr.stats['starttime'])
            npts.append(tr.stats['npts'])
        npts = np.max(np.array(npts))

        A, st = st_to_np_array(st, npts)
        self.options.update(
            {'starttime': starttime,
                'sampling_rate': self.sampling_rate})
        self.logger.debug('Computing Cross-Correlations.')
        A, startlags = self._pxcorr_matrix(A)
        self.logger.debug('Converting Matrix to CorrStream.')
        # put trace into a stream
        cst = CorrStream()
        if A is None:
            # No new data
            return cst
        # Why is that done for every core?
        for ii, (startlag, comb) in enumerate(
                zip(startlags, self.options['combinations'])):
            endlag = startlag + len(A[ii, :])/self.options['sampling_rate']
            cst.append(
                CorrTrace(
                    A[ii], header1=st[comb[0]].stats,
                    header2=st[comb[1]].stats, inv=inv, start_lag=startlag,
                    end_lag=endlag))
        return cst

    def _write(self, cst, tag: str):
        """
        Write correlation stream to files.

        :param cst: CorrStream containing the correlations
        :type cst: :class:`~seismic.correlate.stream.CorrStream`
        """
        if not cst.count():
            self.logger.debug('No new data written.')
            return

        # Make sure that each core writes to a different file
        codelist = list(set(
            [f'{tr.stats.network}.{tr.stats.station}' for tr in cst]))
        # Better if the same cores keep writing to the same files
        codelist.sort()
        # Decide which process writes to which station
        pmap = np.arange(len(codelist))*self.psize/len(codelist)
        pmap = pmap.astype(np.int32)
        ind = pmap == self.rank

        for code in np.array(codelist)[ind]:
            net, stat = code.split('.')
            outf = os.path.join(self.corr_dir, f'{net}.{stat}.h5')
            with CorrelationDataBase(
                    outf, corr_options=self.options) as cdb:
                cdb.add_correlation(
                    cst.select(network=net, station=stat), tag)

    def _generate_data(self) -> Iterator[Tuple[Stream, bool]]:
        """
        Returns an Iterator that loops over each start and end time with the
        requested window length.

        Also will lead to the time windows being prolonged by the sum of
        the length of the tapered end, so that no data is lost. Always
        a hann taper so far. Taper length is 5% of the requested time
        window on each side.

        :yield: An obspy stream containing the time window x for all stations
        that were active during this time.
        :rtype: Iterator[Stream]
        """
        if self.rank == 0:
            # find already available times
            self.ex_dict = self.find_existing_times('subdivision')
            self.logger.info('Already existing data: %s' % str(self.ex_dict))
        else:
            self.ex_dict = None

        self.ex_dict = self.comm.bcast(self.ex_dict, root=0)

        if not self.ex_dict and self.options['preprocess_subdiv']:
            self.options['preprocess_subdiv'] = False
            if self.rank == 0:
                self.logger.warning(
                    'No existing data found.\nAutomatically setting '
                    'preprocess_subdiv to False to optimise performance.')

        # the time window that the loop will go over
        t0 = UTCDateTime(self.options['read_start']).timestamp
        t1 = UTCDateTime(self.options['read_end']).timestamp
        loop_window = np.arange(t0, t1, self.options['read_inc'])

        # Taper ends for the deconvolution
        if self.options['remove_response']:
            tl = 100
        else:
            tl = 0

        # Decide which process reads data from which station
        # Better than just letting one core read as this avoids having to
        # send very big chunks of data using MPI (MPI communication does
        # not support more than 2GB/comm operation)
        pmap = np.arange(len(self.avail_raw_data))*self.psize/len(
            self.avail_raw_data)
        pmap = pmap.astype(np.int32)
        ind = pmap == self.rank
        ind = np.arange(len(self.avail_raw_data))[ind]

        # Loop over read increments
        for t in tqdm(loop_window):
            write_flag = True  # Write length is same as read length
            startt = UTCDateTime(t)
            endt = startt + self.options['read_len']
            st = Stream()
            resp = Inventory()

            # loop over queried stations
            for net, stat, cha in np.array(self.avail_raw_data)[ind]:
                # Load data
                resp.extend(
                    self.store_client.inventory.select(
                        net, stat))
                stext = self.store_client._load_local(
                    net, stat, '*', cha, startt, endt, True, False)
                mu.get_valid_traces(stext)
                if stext is None or not len(stext):
                    # No data for this station to read
                    continue
                st.extend(stext)

            # Stream based preprocessing
            # Downsampling
            # 04/04/2023 Downsample before preprocessing for performance
            # Check sampling frequency
            sampling_rate = self.options['sampling_rate']
            # AA-Filter is done in this function as well
            st = mu.resample_or_decimate(st, sampling_rate)
            # The actual data in the mseeds was changed from int to float64
            # now,
            # Save some space by changing it back to 32 bit (most of the
            # digitizers work at 24 bit anyways)
            mu.stream_require_dtype(st, np.float32)

            if not self.options['preprocess_subdiv']:
                try:
                    self.logger.debug('Preprocessing stream...')
                    st = preprocess_stream(
                        st, self.store_client, resp, startt, endt, tl,
                        **self.options)
                except ValueError as e:
                    self.logger.error(
                        'Stream preprocessing failed for '
                        f'{st[0].stats.network}.{st[0].stats.station} and time'
                        f' {t}.\nThe Original Error Message was {e}.')
                    continue

            # Slice the stream in correlation length
            # -> Loop over correlation increments
            for ii, win in enumerate(generate_corr_inc(st, **self.options)):
                winstart = startt + ii*self.options['subdivision']['corr_inc']
                winend = winstart + self.options['subdivision']['corr_len']

                # Gather time windows from all stations to all cores
                winl = self.comm.allgather(win)
                win = Stream()
                for winp in winl:
                    win.extend(winp)
                win.sort(keys=['network', 'station', 'channel'])

                # Get correlation combinations
                if self.rank == 0:
                    self.logger.debug('Calculating combinations...')
                    self.options['combinations'] = calc_cross_combis(
                        win, self.ex_dict, self.options['combination_method'],
                        rcombis=self.rcombis)
                else:
                    self.options['combinations'] = None
                self.options['combinations'] = self.comm.bcast(
                    self.options['combinations'], root=0)

                if not len(self.options['combinations']):
                    # no new combinations for this time period
                    self.logger.info(
                        f'No new data for times {winstart}-{winend}')
                    continue
                # Remove traces that won't be accessed at all
                win_indices = np.arange(len(win))
                combindices = np.unique(
                    np.hstack(self.options['combinations']))
                popindices = np.flip(
                    np.setdiff1d(win_indices, combindices))
                for popi in popindices:
                    del win[popi]
                if len(popindices):
                    # now we have to recompute the combinations
                    if self.rank == 0:
                        self.logger.debug('removing redundant data.')
                        self.logger.debug('Recalculating combinations...')
                        self.options['combinations'] = calc_cross_combis(
                            win, self.ex_dict,
                            self.options['combination_method'],
                            rcombis=self.rcombis)
                    else:
                        self.options['combinations'] = None
                    self.options['combinations'] = self.comm.bcast(
                        self.options['combinations'], root=0)

                    if not len(self.options['combinations']):
                        # no new combinations for this time period
                        self.logger.info(
                            f'No new data for times {winstart}-{winend}')
                        continue
                # Stream based preprocessing
                if self.options['preprocess_subdiv']:
                    try:
                        win = preprocess_stream(
                            win, self.store_client, resp, winstart, winend,
                            tl, **self.options)
                    except ValueError as e:
                        self.logger.error(
                            'Stream preprocessing failed for '
                            f'{st[0].stats.network}.{st[0].stats.station}'
                            ' and time '
                            f'{t}.\nThe Original Error Message was {e}.')
                        continue
                    if self.rank == 0:
                        self.options['combinations'] = calc_cross_combis(
                            win, self.ex_dict,
                            self.options['combination_method'],
                            rcombis=self.rcombis)
                    else:
                        self.options['combinations'] = None
                    self.options['combinations'] = self.comm.bcast(
                        self.options['combinations'], root=0)

                if not len(win):
                    # no new combinations for this time period
                    self.logger.info(
                        f'No new data for times {winstart}-{winend}')
                    continue

                self.logger.debug('Working on correlation times %s-%s' % (
                    str(win[0].stats.starttime), str(win[0].stats.endtime)))
                yield win, write_flag
                write_flag = False

    def _pxcorr_matrix(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # time domain processing
        # map of traces on processes
        ntrc = A.shape[0]
        pmap = np.arange(ntrc)*self.psize/ntrc
        # This step was not in the original but is necessary for it to work?
        # maybe a difference in an old python/np version?
        pmap = pmap.astype(np.int32)

        # indices for traces to be worked on by each process
        ind = pmap == self.rank

    ######################################
        corr_args = self.options['corr_args']
        # time domain pre-processing
        params = {}
        for key in list(corr_args.keys()):
            if 'Processing' not in key:
                params.update({key: corr_args[key]})
        params['sampling_rate'] = self.sampling_rate
        # The steps that aren't done before

        # nans from the masked parts are set to 0
        np.nan_to_num(A, copy=False)

        for proc in corr_args['TDpreProcessing']:
            func = func_from_str(proc['function'])
            A[ind, :] = func(A[ind, :], proc['args'], params)

        # zero-padding
        A = pptd.zeroPadding(A, {'type': 'avoidWrapFastLen'}, params)

        ######################################
        # FFT
        # Allocate space for rfft of data
        zmsize = A.shape

        # use next fast len instead?
        fftsize = zmsize[1]//2+1
        B = np.zeros((ntrc, fftsize), dtype=np.csingle)

        B[ind, :] = np.fft.rfft(A[ind, :], axis=1)

        freqs = np.fft.rfftfreq(zmsize[1], 1./self.sampling_rate)

        ######################################
        # frequency domain pre-processing
        params.update({'freqs': freqs})
        # Here, I will have to make sure to add all the functions to the module
        for proc in corr_args['FDpreProcessing']:

            # The big advantage of this rather lengthy code is that we can also
            # import any function that has been defined anywhere else (i.e,
            # not only within the miic framework)
            func = func_from_str(proc['function'])
            B[ind, :] = func(B[ind, :], proc['args'], params)

        ######################################
        # collect results
        self.comm.Allreduce(MPI.IN_PLACE, [B, MPI.FLOAT], op=MPI.SUM)

        ######################################
        # correlation
        csize = len(self.options['combinations'])
        irfftsize = (fftsize-1)*2
        sampleToSave = int(
            np.ceil(
                corr_args['lengthToSave'] * self.sampling_rate))
        C = np.zeros((csize, sampleToSave*2+1), dtype=np.float32)

        pmap = np.arange(csize)*self.psize/csize
        pmap = pmap.astype(np.int32)
        ind = pmap == self.rank
        ind = np.arange(csize)[ind]
        startlags = np.zeros(csize, dtype=np.float32)
        for ii in ind:
            # offset of starttimes in samples(just remove fractions of samples)
            offset = (
                self.options['starttime'][
                    self.options['combinations'][ii][0]] - self.options[
                        'starttime'][self.options['combinations'][ii][1]])
            if corr_args['center_correlation']:
                roffset = 0.
            else:
                # offset exceeding a fraction of integer
                roffset = np.fix(
                    offset * self.sampling_rate) / self.sampling_rate
            # faction of samples to be compenasated by shifting
            offset -= roffset
            # normalization factor of fft correlation
            if corr_args['normalize_correlation']:
                norm = (
                    np.sqrt(
                        2.*np.sum(B[self.options[
                            'combinations'][ii][0], :]
                            * B[self.options['combinations'][ii][0], :].conj())
                        - B[self.options['combinations'][ii][0], 0]**2)
                    * np.sqrt(
                        2.*np.sum(B[self.options[
                            'combinations'][ii][1], :]
                            * B[self.options['combinations'][ii][1], :].conj())
                        - B[self.options['combinations'][ii][1], 0]**2)
                    / irfftsize).real
            else:
                norm = 1.

            M = (
                B[self.options['combinations'][ii][0], :].conj()
                * B[self.options['combinations'][ii][1], :]
                * np.exp(1j * freqs * offset * 2 * np.pi))

            ######################################
            # frequency domain postProcessing
            #
            tmp = np.fft.irfft(M).real

            # cut the center and do fftshift
            C[ii, :] = np.concatenate(
                (tmp[-sampleToSave:], tmp[:sampleToSave+1]))/norm
            startlags[ii] = - sampleToSave / self.sampling_rate \
                - roffset

        ######################################
        # time domain postProcessing

        ######################################
        # collect results
        self.logger.debug('%s %s' % (C.shape, C.dtype))
        self.logger.debug('combis: %s' % (self.options['combinations']))

        self.comm.Allreduce(MPI.IN_PLACE, [C, MPI.FLOAT], op=MPI.SUM)
        self.comm.Allreduce(
            MPI.IN_PLACE, [startlags, MPI.FLOAT], op=MPI.SUM)

        return (C, startlags)


def st_to_np_array(st: Stream, npts: int) -> Tuple[np.ndarray, Stream]:
    """
    Converts an obspy stream to a matrix with the shape (npts, st.count()).
    Also returns the same stream but without the data arrays in tr.data.

    :param st: Input Stream
    :type st: Stream
    :param npts: Maximum number of samples per Trace
    :type npts: int
    :return: A stream and a matrix
    :rtype: np.ndarray
    """
    A = np.zeros((st.count(), npts), dtype=np.float32)
    for ii, tr in enumerate(st):
        A[ii, :tr.stats.npts] = tr.data
        del tr.data  # Not needed any more, just uses up RAM
    return A, st


def _compare_existing_data(ex_corr: dict, tr0: Trace, tr1: Trace) -> bool:
    # The actual starttime for the header is the later one of the two
    net0 = tr0.stats.network
    stat0 = tr0.stats.station
    cha0 = tr0.stats.channel
    net1 = tr1.stats.network
    stat1 = tr1.stats.station
    cha1 = tr1.stats.channel
    # Probably faster than checking a huge dict twice
    flip = ([net0, net1], [stat0, stat1], [cha0, cha1]) \
        != sort_comb_name_alphabetically(
        net0, stat0, net1, stat1, cha0, cha1)
    corr_start = max(tr0.stats.starttime, tr1.stats.starttime)
    try:
        if flip:
            return corr_start.format_fissures() in ex_corr[
                f'{net1}.{stat1}'][f'{net0}.{stat0}'][
                '%s-%s' % (
                    tr1.stats.channel, tr0.stats.channel)]
        else:
            return corr_start.format_fissures() in ex_corr[
                f'{net0}.{stat0}'][f'{net1}.{stat1}'][
                '%s-%s' % (
                    tr0.stats.channel, tr1.stats.channel)]
    except KeyError:
        return False


def calc_cross_combis(
    st: Stream, ex_corr: dict, method: str = 'betweenStations',
        rcombis: List[str] = None) -> list:
    """
    Calculate a list of all cross correlation combination
    of traces in the stream: i.e. all combination with two different
    stations involved.

    :param st: Stream holding the tracecs to be correlated
    :type st: :class:`~obspy.Stream`
    :param ex_corr: dict holding the correlations that already exist in db
    :type ex_corr: dict
    :type method: stringf
    :param method: Determines which traces of the strem are combined.
    :param rcombis: requested combinations, only works if
        `method==betweenStations`.
    :type rcombis: List[str] strings are in form net0-net1.stat0-stat1

        ``'betweenStations'``:
            Traces are combined if either their station or
            their network names are different.
        ``'betweenComponents'``:
            Traces are combined if their components (last
            letter of channel name) names are different and their station and
            network names are identical (single station cross-correlation).
        ``'autoComponents'``:
            Traces are combined only with themselves.
        ``'allSimpleCombinations'``:
            All Traces are combined once (onle one of
            (0,1) and (1,0))
        ``'allCombinations'``:
            All traces are combined in both orders ((0,1) and (1,0))
    """

    combis = []
    # sort alphabetically
    st.sort(keys=['network', 'station', 'channel'])
    if method == 'betweenStations':
        for ii, tr in enumerate(st):
            for jj in range(ii+1, len(st)):
                tr1 = st[jj]
                n = tr.stats.network
                n2 = tr1.stats.network
                s = tr.stats.station
                s2 = tr1.stats.station
                if n != n2 or s != s2:
                    # check first whether this combi is in dict
                    if _compare_existing_data(ex_corr, tr, tr1):
                        continue
                    if rcombis is not None and not any(all(
                        i0 in i1 for i0 in [
                            n, n2, s, s2]) for i1 in rcombis):
                        # If particular combis are requested, compute only
                        # those
                        continue
                    combis.append((ii, jj))
    elif method == 'betweenComponents':
        for ii, tr in enumerate(st):
            for jj in range(ii+1, len(st)):
                tr1 = st[jj]
                if ((tr.stats['network'] == tr1.stats['network'])
                    and (tr.stats['station'] == tr1.stats['station'])
                    and (
                        tr.stats['channel'][-1] != tr1.stats['channel'][-1])):
                    if _compare_existing_data(ex_corr, tr, tr1):
                        continue
                    combis.append((ii, jj))
    elif method == 'autoComponents':
        for ii, tr in enumerate(st):
            if _compare_existing_data(ex_corr, tr, tr):
                continue
            combis.append((ii, ii))
    elif method == 'allSimpleCombinations':
        for ii, tr in enumerate(st):
            for jj in range(ii, len(st)):
                tr1 = st[jj]
                if _compare_existing_data(ex_corr, tr, tr1):
                    continue
                combis.append((ii, jj))
    elif method == 'allCombinations':
        for ii, tr in enumerate(st):
            for jj, tr1 in enumerate(st):
                if _compare_existing_data(ex_corr, tr, tr1):
                    continue
                combis.append((ii, jj))
    else:
        raise ValueError("Method has to be one of ('betweenStations', "
                         "'betweenComponents', 'autoComponents', "
                         "'allSimpleCombinations' or 'allCombinations').")
    if not len(combis):
        warn('Method %s found no combinations.' % method)
    return combis


# All the rotations are still untested, should do that at some point

# def rotate_multi_corr_stream(st: Stream) -> Stream:
#     """Rotate a stream with full Greens tensor from ENZ to RTZ

#     Take a stream with numerous correlation traces and rotate the
#     combinations of ENZ components into combinations of RTZ components in
#     all nine components of the Green's tensor are present. If not all nine
#     components are present no trace for this station combination is returned.

#     :type st: obspy.stream
#     :param st: stream with data in ENZ system
#     :rtype: obspy.stream
#     :return: stream in the RTZ system
#     """

#     out_st = Stream()
#     while st:
#         tl = list(range(9))
#         tst = st.select(network=st[0].stats['network'],
#                         station=st[0].stats['station'])
#         cnt = 0
#         for ttr in tst:
#             if ttr.stats['channel'][2] == 'E':
#                 if ttr.stats['channel'][6] == 'E':
#                     tl[0] = ttr
#                     cnt += 1
#                 elif ttr.stats['channel'][6] == 'N':
#                     tl[1] = ttr
#                     cnt += 2
#                 elif ttr.stats['channel'][6] == 'Z':
#                     tl[2] = ttr
#                     cnt += 4
#             elif ttr.stats['channel'][2] == 'N':
#                 if ttr.stats['channel'][6] == 'E':
#                     tl[3] = ttr
#                     cnt += 8
#                 elif ttr.stats['channel'][6] == 'N':
#                     tl[4] = ttr
#                     cnt += 16
#                 elif ttr.stats['channel'][6] == 'Z':
#                     tl[5] = ttr
#                     cnt += 32
#             elif ttr.stats['channel'][2] == 'Z':
#                 if ttr.stats['channel'][6] == 'E':
#                     tl[6] = ttr
#                     cnt += 64
#                 elif ttr.stats['channel'][6] == 'N':
#                     tl[7] = ttr
#                     cnt += 128
#                 elif ttr.stats['channel'][6] == 'Z':
#                     tl[8] = ttr
#                     cnt += 256
#         if cnt == 2**9-1:
#             st0 = Stream()
#             for t in tl:
#                 st0.append(t)
#             st1 = _rotate_corr_stream(st0)
#             out_st += st1
#         elif cnt == 27:  # only horizontal component combinations present
#             st0 = Stream()
#             for t in [0, 1, 3, 4]:
#                 st0.append(tl[t])
#             st1 = _rotate_corr_stream_horizontal(st0)
#             out_st += st1
#         elif cnt == 283:  # horizontal combinations + ZZ
#             st0 = Stream()
#             for t in [0, 1, 3, 4]:
#                 st0.append(tl[t])
#             st1 = _rotate_corr_stream_horizontal(st0)
#             out_st += st1
#             out_st.append(tl[8])
#         for ttr in tst:
#             for ind, tr in enumerate(st):
#                 if ttr.id == tr.id:
#                     st.pop(ind)

#     return out_st


# def _rotate_corr_stream_horizontal(st: Stream) -> Stream:
#     """ Rotate traces in stream from the EE-EN-NE-NN system to
#     the RR-RT-TR-TT system. The letters give the component order
#     in the input and output streams. Input traces are assumed to be of same
#     length and simultaneously sampled.
#     """

#     # rotation angles
#     # phi1 : counter clockwise angle between E and R(towards second station)
#     # the leading -1 accounts fact that we rotate the coordinate system,
#     # not a vector
#     phi1 = - np.pi/180*(90-st[0].stats['sac']['az'])
#     # phi2 : counter clockwise angle between E and R(away from first station)
#     phi2 = - np.pi/180*(90-st[0].stats['sac']['baz']+180)

#     c1 = np.cos(phi1)
#     s1 = np.sin(phi1)
#     c2 = np.cos(phi2)
#     s2 = np.sin(phi2)

#     rt = Stream()
#     RR = st[0].copy()
#     RR.data = c1*c2*st[0].data - c1*s2*st[1].data - s1*c2*st[2].data +\
#         s1*s2*st[3].data
#     tcha = list(RR.stats['channel'])
#     tcha[2] = 'R'
#     tcha[6] = 'R'
#     RR.stats['channel'] = ''.join(tcha)
#     rt.append(RR)

#     RT = st[0].copy()
#     RT.data = c1*s2*st[0].data + c1*c2*st[1].data - s1*s2*st[2].data -\
#         s1*c2*st[3].data
#     tcha = list(RT.stats['channel'])
#     tcha[2] = 'R'
#     tcha[6] = 'T'
#     RT.stats['channel'] = ''.join(tcha)
#     rt.append(RT)

#     TR = st[0].copy()
#     TR.data = s1*c2*st[0].data - s1*s2*st[1].data + c1*c2*st[2].data -\
#         c1*s2*st[3].data
#     tcha = list(TR.stats['channel'])
#     tcha[2] = 'T'
#     tcha[6] = 'R'
#     TR.stats['channel'] = ''.join(tcha)
#     rt.append(TR)

#     TT = st[0].copy()
#     TT.data = s1*s2*st[0].data + s1*c2*st[1].data + c1*s2*st[2].data +\
#         c1*c2*st[3].data
#     tcha = list(TT.stats['channel'])
#     tcha[2] = 'T'
#     tcha[6] = 'T'
#     TT.stats['channel'] = ''.join(tcha)
#     rt.append(TT)

#     return rt


# def _rotate_corr_stream(st: Stream) -> Stream:
#     """ Rotate traces in stream from the EE-EN-EZ-NE-NN-NZ-ZE-ZN-ZZ system to
#     the RR-RT-RZ-TR-TT-TZ-ZR-ZT-ZZ system. The letters give the component
#     in the input and output streams. Input traces are assumed to be of same
#     length and simultaneously sampled.
#     """

#     # rotation angles
#     # phi1 : counter clockwise angle between E and R(towards second station)
#     # the leading -1 accounts fact that we rotate the coordinate system,
#     # not a vector
#     phi1 = - np.pi/180*(90-st[0].stats['sac']['az'])
#     # phi2 : counter clockwise angle between E and R(away from first station)
#     phi2 = - np.pi/180*(90-st[0].stats['sac']['baz']+180)

#     c1 = np.cos(phi1)
#     s1 = np.sin(phi1)
#     c2 = np.cos(phi2)
#     s2 = np.sin(phi2)

#     rtz = Stream()
#     RR = st[0].copy()
#     RR.data = c1*c2*st[0].data - c1*s2*st[1].data - s1*c2*st[3].data +\
#         s1*s2*st[4].data
#     tcha = list(RR.stats['channel'])
#     tcha[2] = 'R'
#     tcha[6] = 'R'
#     RR.stats['channel'] = ''.join(tcha)
#     rtz.append(RR)

#     RT = st[0].copy()
#     RT.data = c1*s2*st[0].data + c1*c2*st[1].data - s1*s2*st[3].data -\
#         s1*c2*st[4].data
#     tcha = list(RT.stats['channel'])
#     tcha[2] = 'R'
#     tcha[6] = 'T'
#     RT.stats['channel'] = ''.join(tcha)
#     rtz.append(RT)

#     RZ = st[0].copy()
#     RZ.data = c1*st[2].data - s1*st[5].data
#     tcha = list(RZ.stats['channel'])
#     tcha[2] = 'R'
#     tcha[6] = 'Z'
#     RZ.stats['channel'] = ''.join(tcha)
#     rtz.append(RZ)

#     TR = st[0].copy()
#     TR.data = s1*c2*st[0].data - s1*s2*st[1].data + c1*c2*st[3].data -\
#         c1*s2*st[4].data
#     tcha = list(TR.stats['channel'])
#     tcha[2] = 'T'
#     tcha[6] = 'R'
#     TR.stats['channel'] = ''.join(tcha)
#     rtz.append(TR)

#     TT = st[0].copy()
#     TT.data = s1*s2*st[0].data + s1*c2*st[1].data + c1*s2*st[3].data +\
#         c1*c2*st[4].data
#     tcha = list(TT.stats['channel'])
#     tcha[2] = 'T'
#     tcha[6] = 'T'
#     TT.stats['channel'] = ''.join(tcha)
#     rtz.append(TT)

#     TZ = st[0].copy()
#     TZ.data = s1*st[2].data + c1*st[5].data
#     tcha = list(TZ.stats['channel'])
#     tcha[2] = 'T'
#     tcha[6] = 'Z'
#     TZ.stats['channel'] = ''.join(tcha)
#     rtz.append(TZ)

#     ZR = st[0].copy()
#     ZR.data = c2*st[6].data - s2*st[7].data
#     tcha = list(ZR.stats['channel'])
#     tcha[2] = 'Z'
#     tcha[6] = 'R'
#     ZR.stats['channel'] = ''.join(tcha)
#     rtz.append(ZR)

#     ZT = st[0].copy()
#     ZT.data = s2*st[6].data + c2*st[7].tuple
#     ZT.stats['channel'] = ''.join(tcha)
#     rtz.append(ZT)

#     rtz.append(st[8].copy())

#     return rtz


def sort_comb_name_alphabetically(
    network1: str, station1: str, network2: str, station2: str,
    channel1: Optional[str] = '',
        channel2: Optional[str] = '') -> Tuple[
        list, list]:
    """
    Returns the alphabetically sorted network and station codes from the two
    station.

    :param network1: network code of first station
    :type network1: str
    :param station1: station code of first station
    :type station1: str
    :param network2: Network code of second station
    :type network2: str
    :param station2: Station Code of second Station
    :type station2: str
    :return: A tuple containing the list of the network codes sorted
        and the list of the station codes sorted.
    :rtype: Tuple[ list, list]

    Examples
    --------
    >>> net1 = 'IU'  # Network Code of first station
    >>> stat1 = 'HRV'  # Station Code of first station
    >>> net2 = 'XN'
    >>> stat2 = 'NEP06'
    >>> print(sort_comb_name_aphabetically(
            net1, stat1, net2, stat2))
    (['IU', 'XN'], ['HRV', 'NEP06'])
    >>> print(sort_comb_name_aphabetically(
            net2, stat2, net1, stat1))
    (['IU', 'XN'], ['HRV', 'NEP06'])
    >>> # Different combination
    >>> net1 = 'YP'  # Network Code of first station
    >>> stat1 = 'AB3'  # Station Code of first station
    >>> net2 = 'XN'
    >>> stat2 = 'NEP06'
    >>> print(sort_comb_name_aphabetically(
            net1, stat1, net2, stat2))
    (['XN', 'YP'], ['NEP06', 'AB3'])
    >>> # Different combination
    >>> net1 = 'XN'  # Network Code of first station
    >>> stat1 = 'NEP07'  # Station Code of first station
    >>> net2 = 'XN'
    >>> stat2 = 'NEP06'
    >>> print(sort_comb_name_aphabetically(
            net1, stat1, net2, stat2))
    (['XN', 'XN'], ['NEP06', 'NEP07'])
    """
    if not all([isinstance(arg, str) for arg in [
            network1, network2, station1, station2]]):
        raise TypeError('All arguments have to be strings.')
    sort1 = network1 + station1 + channel1
    sort2 = network2 + station2 + channel2
    sort = [sort1, sort2]
    sorted = sort.copy()
    sorted.sort()
    if sort == sorted:
        netcomb = [network1, network2]
        statcomb = [station1, station2]
        chacomb = [channel1, channel2]
    else:
        netcomb = [network2, network1]
        statcomb = [station2, station1]
        chacomb = [channel2, channel1]
    return netcomb, statcomb, chacomb


def compute_network_station_combinations(
    netlist: list, statlist: list,
    method: str = 'betweenStations', combis: List[str] = None) -> Tuple[
        list, list]:
    """
    Return the network and station codes of the correlations for the provided
    lists of networks and stations and the queried combination method.

    :param netlist: List of network codes
    :type netlist: list
    :param statlist: List of Station Codes
    :type statlist: list
    :param method: The combination method to use. Has to be one of the
        following: `betweenStation`, `betweenComponents`, `autoComponents`,
        `allSimpleCombinations`, or `allCombinations`,
        defaults to 'betweenStations'.
    :type method: str, optional
    :param combis: List of desired station combinations.
        Given as [net0-net1.stat0-stat1]. Optional.
    :type combis: List[str]
    :raises ValueError: for unkown combination methods.
    :return: A tuple containing the list of the correlation network code
        and the list of the correlation station code.
    :rtype: Tuple[list, list]
    """
    netcombs = []
    statcombs = []
    if method == 'betweenStations':
        for ii, (n, s) in enumerate(zip(netlist, statlist)):
            for jj in range(ii+1, len(netlist)):
                n2 = netlist[jj]
                s2 = statlist[jj]
                if n != n2 or s != s2:
                    nc, sc, _ = sort_comb_name_alphabetically(n, s, n2, s2)
                    # Check requested combinations
                    if (
                        combis is not None
                        and f'{nc[0]}-{nc[1]}.{sc[0]}-{sc[1]}' not in combis
                    ):
                        continue
                    netcombs.append('%s-%s' % (nc[0], nc[1]))
                    statcombs.append('%s-%s' % (sc[0], sc[1]))

    elif method == 'betweenComponents' or method == 'autoComponents':
        netcombs = [n+'-'+n for n in netlist]
        statcombs = [s+'-'+s for s in statlist]
    elif method == 'allSimpleCombinations':
        for ii, (n, s) in enumerate(zip(netlist, statlist)):
            for jj in range(ii, len(netlist)):
                n2 = netlist[jj]
                s2 = statlist[jj]
                nc, sc, _ = sort_comb_name_alphabetically(n, s, n2, s2)
                netcombs.append('%s-%s' % (nc[0], nc[1]))
                statcombs.append('%s-%s' % (sc[0], sc[1]))
    elif method == 'allCombinations':
        for n, s in zip(netlist, statlist):
            for n2, s2 in zip(netlist, statlist):
                nc, sc, _ = sort_comb_name_alphabetically(n, s, n2, s2)
                netcombs.append('%s-%s' % (nc[0], nc[1]))
                statcombs.append('%s-%s' % (sc[0], sc[1]))
    else:
        raise ValueError("Method has to be one of ('betweenStations', "
                         "'betweenComponents', 'autoComponents', "
                         "'allSimpleCombinations' or 'allCombinations').")
    return netcombs, statcombs


def preprocess_stream(
    st: Stream, store_client: Store_Client, inv: Inventory or None,
    startt: UTCDateTime, endt: UTCDateTime, taper_len: float,
    remove_response: bool, subdivision: dict,
    preProcessing: List[dict] = None,
        **kwargs) -> Stream:
    """
    Does the preprocessing on a per stream basis. Most of the parameters can be
    fed in by using the "yaml" dict as kwargs.

    :param st: Input Stream to be processed
    :type st: :class:`obspy.core.stream.Stream`
    :param store_client: Store Client for the database
    :type store_client: :class:`~seismic.trace_data.waveform.Store_Client`
    :param inv: Station response, can be None if ``remove_response=False``.
    :type inv: Inventory or None
    :param startt: Starttime that the stream should be clipped / padded to.
    :type startt: :class:`obspy.UTCDateTime`
    :param endt: Endtime that the stream should be clipped / padded to.
    :type endt: :class:`obspy.UTCDateTime`
    :param taper_len: If the instrument response is removed, one might want to
        taper to mitigate the acausal effects of the deconvolution. This is
        the length of such a taper in seconds.
    :type taper_len: float
    :param remove_response: Should the instrument response be removed?
    :type remove_response: bool
    :param subdivision: Dictionary holding information about the correlation
        lenghts and increments.
    :type subdivision: dict
    :param preProcessing: List holding information about the different external
        preprocessing functions to be applied, defaults to None
    :type preProcessing: List[dict], optional
    :raises ValueError: For sampling rates higher than the stream's native
        sampling rate (upsampling is not permitted).
    :return: The preprocessed stream.
    :rtype: :class:`obspy.core.stream.Stream`
    """
    if not st.count():
        return st
    # To deal with any nans/masks
    st = st.split()
    st.sort(keys=['starttime'])

    # Clip to these again to remove the taper
    old_starts = [deepcopy(tr.stats.starttime) for tr in st]
    old_ends = [deepcopy(tr.stats.endtime) for tr in st]

    if remove_response:
        # taper before instrument response removal
        if taper_len:
            st = ppst.cos_taper_st(st, taper_len, False, True)
        try:
            if inv:
                ninv = inv
                st.attach_response(ninv)
            st.remove_response(taper=False)  # Changed for testing purposes
        except ValueError:
            print('Station response not found ... loading from remote.')
            # missing station response
            ninv = store_client.rclient.get_stations(
                network=st[0].stats.network, station=st[0].stats.station,
                channel='*', level='response')
            st.attach_response(ninv)
            st.remove_response(taper=False)
            store_client._write_inventory(ninv)

    # Sometimes Z has reversed polarity
    if inv:
        try:
            mu.correct_polarity(st, inv)
        except Exception as e:
            print(e)

    mu.discard_short_traces(st, subdivision['corr_len']/20)

    if preProcessing:
        for procStep in preProcessing:
            func = func_from_str(procStep['function'])
            st = func(st, **procStep['args'])
    # Remove the artificial taper from earlier
    for tr, ostart, oend in zip(st, old_starts, old_ends):
        tr.trim(starttime=ostart, endtime=oend)
    st.merge()
    st.trim(startt, endt, pad=True)

    mu.discard_short_traces(st, subdivision['corr_len']/20)
    return st


def generate_corr_inc(
    st: Stream, subdivision: dict, read_len: int,
        **kwargs) -> Iterator[Stream]:
    """
    Subdivides the preprocessed streams into parts of equal length using
    the parameters ``cor_inc`` and ``cor_len`` in ``subdivision``.
    This function can be acessed by several processes in parallel

    :param st: The preprocessed input stream
    :type st: :class:`obspy.core.stream.Stream`
    :param subdivision: Dictionary holding the information about the
        correlation length and increment.
    :type subdivision: dict
    :param read_len: Length to be read from disk in seconds
    :type read_len: int
    :yield: Equal length windows, padded with nans / masked if data is missing.
    :rtype: Generator[Stream]
    """

    try:
        # Second loop to return the time window in correlation length
        for ii, win0 in enumerate(st.slide(
            subdivision['corr_len']-st[0].stats.delta, subdivision['corr_inc'],
                include_partial_windows=True)):

            # We use trim so the windows have the right time and
            # are filled with masked arrays for places without values
            starttrim = st[0].stats.starttime + ii*subdivision['corr_inc']
            endtrim = st[0].stats.starttime + ii*subdivision['corr_inc'] +\
                subdivision['corr_len']-st[0].stats.delta
            win = win0.trim(starttrim, endtrim, pad=True)
            mu.get_valid_traces(win)

            yield win

    except IndexError:
        # processes with no data end up here
        win = Stream()
        # A little dirty, but it has to go through an equally long loop
        # else this will cause a deadlock
        for _ in range(int(np.ceil(
                read_len/subdivision['corr_inc']))):
            yield win
