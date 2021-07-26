'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 3rd June 2021 04:15:57 pm
Last Modified: Monday, 26th July 2021 09:19:22 am
'''
import logging
import os
from typing import Tuple
import yaml

from mpi4py import MPI
import numpy as np
from obspy import UTCDateTime

from miic3.db.corr_hdf5 import CorrelationDataBase
from miic3.utils.miic_utils import log_lvl
# 12.07 Not needed here anymore, I am 100% sure that this produces exactly
# the same results as the old miic
# from miic3.utils.io import corrmat_to_corrbulk, load_corrbulk_from_mat


class Monitor(object):
    def __init__(self, options: dict or str):
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
            "miic3.monitor.Monitor0%s" % str(self.rank))
        self.logger.setLevel(loglvl)

        # also catch the warnings
        logging.captureWarnings(True)
        warnlog = logging.getLogger('py.warnings')
        fh = logging.FileHandler(os.path.join(logdir, 'monitor%srank0%s' % (
            tstr, self.rank)))
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
            :func:`~miic3.monitor.monitor.make_time_list`
        """
        starttimes, endtimes = make_time_list(
            self.options['dv']['start_date'], self.options['dv']['end_date'],
            self.options['dv']['date_inc'], self.options['dv']['win_len'])
        return starttimes, endtimes

    def _find_available_corrs(self):
        netlist = []
        statlist = []
        infiles = []
        for f in os.listdir(self.indir):
            split = f.split('.')
            netlist.append(split[0])
            statlist.append(split[1])
            infiles.append(os.path.join(self.indir, f))
        self.logger.info('Found correlation data for the following station \
and network combinations %s' % str(
                ['{n}.{s}'.format(n=n, s=s) for n, s in zip(
                    netlist, statlist)]))
        return netlist, statlist, infiles

    def compute_velocity_change(
        self, corr_file: str, tag: str, network: str, station: str,
            channel: str):
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
        cb.resample(self.starttimes, self.endtimes)
        cb.filter(
            (self.options['dv']['freq_min'], self.options['dv']['freq_max']))
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
        # Decide whether to use already stacked data
        if self.options['co']['subdivision']['recombine_subdivision'] and\
            self.options['co']['read_inc'] \
                <= self.options['dv']['date_inc']:
            tag = 'recombined'
        else:
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
        for ii in ind:
            corr_file, net, stat, cha = plist[ii]
            try:
                self.compute_velocity_change(
                    corr_file, tag, net, stat, cha)
            except Exception as e:
                self.logger.error(e)


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

    ..note:: see `obspy's documentation <https://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html>`
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
