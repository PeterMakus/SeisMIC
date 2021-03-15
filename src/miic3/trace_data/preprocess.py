'''
A module to create seismic ambient noise correlations.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 4th March 2021 03:54:06 pm
Last Modified: Monday, 15th March 2021 10:23:27 am
'''
from collections import namedtuple
from glob import glob
import os
from typing import NamedTuple
from warnings import warn

from joblib import Parallel, delayed, cpu_count
import numpy as np
from obspy import UTCDateTime, Stream
from pyasdf.asdf_data_set import ASDFDataSet

from .waveform import Store_Client


# Class to define filter frequencies
Filter = namedtuple('Filter', 'lowcut highcut')


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class FrequencyError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class Preprocessor(object):
    def __init__(
        self, store_client: Store_Client, filter: tuple,
            sampling_frequency: float) -> None:
        super().__init__()
        assert sampling_frequency < 2*filter[1], \
            "The highcut frequency of the filter has to be lower than the \
                Nyquist frequency (sampling frequency/2) of the signal to \
                    prevent aliasing. Current Nyquist frequency is: %s."\
                         % sampling_frequency/2
        self.store_client = store_client
        self.filter = Filter(filter[0], filter[1])
        self.sampling_frequency = sampling_frequency

    def preprocess_bulk(
        self, network: str or None = None,
        starttime: UTCDateTime or None = None,
            endtime: UTCDateTime or None = None):
        Parallel(n_jobs=-1)(
                    delayed(self.preprocess)(
                        network, station, '*', '*', starttime, endtime)
                    for network, station in self._all_stations_raw(network))

    def preprocess(
        self, network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime or None = None,
            endtime: UTCDateTime or None = None):

        assert '*' not in station or type(station) != list, 'To process data\
             from several stations, use Correlator.preprocess_bulk!'

        if not starttime or not endtime:
            warn(
                'Returned start and endtimes will not be checked due to' +
                ' wildcard.', category=UserWarning)
            _check_times = False
            starttime, endtime = self.store_client._get_times(network, station)
            if not starttime:
                # No data
                return
        else:
            _check_times = True

        # Save some memory: will cut the requests in shorter chunks.
        # Else this becomes problematic especially, when processing with
        # several threads at the same time and the very long streams have to be
        # held in RAM.
        if endtime-starttime > 3600*12:
            starttimes = []
            endtimes = []
            while starttime < endtime:
                starttimes.append(starttime)
                starttime = starttime+1800*12
                endtimes.append(starttime)
        else:
            starttimes = [starttime]
            endtimes = [endtime]

        for starttime, endtime in zip(starttimes, endtimes):
            # Return obspy stream with data from this station
            st = self.store_client.get_waveforms(
                network, station, location, channel, starttime, endtime,
                _check_times=_check_times)

            try:
                st, resp = self._preprocess(st)
            except FrequencyError as e:
                warn(e + ' Trace is skipped.')
                continue

            # Create folder if it does not exist
            os.makedirs(
                os.path.join(self.store_client.sds_root, 'preprocessed'),
                exist_ok=True)

            with ASDFDataSet(
                os.path.join(
                    self.store_client.sds_root, 'preprocessed',
                    '%s.%s.h5' % (network, station))) as ds:
                ds.add_waveforms(st, tag='processed')
                ds.add_stationxml(resp)

    def _all_stations_raw(
            self, network: str or None = None) -> list:
        """
        Returns a list of stations for which raw data is available.
        Very similar to the lClient.get_all_stations() method with the
        difference that this one allows to filter for particular networks.

        :param network: Only return stations from this network. `network==None`
        is similar to using a wildcard. Defaults to None
        :type network: strorNone, optional
        :return: List of network and station codes in the form:
        `[[net0,stat0], [net1,stat1]]`.
        :rtype: list
        """

        # If no network is defined use all
        network = network or '*'
        oslist = glob(
            os.path.join(self.store_client.sds_root, '*', network, '*'))
        statlist = []
        for path in oslist:
            # Add all network and station combinations to list
            code = path.split('/')[-2:]
            if code not in statlist:
                statlist.append(code)
        return statlist

    def _preprocess(self, st: Stream) -> Stream:
        # Check sampling frequency
        if self.filter.highcut > st[0].stats.sampling_frequency/2:
            raise FrequencyError(
                'The highcut frequency of the filter (%sHz) is higher than the\
                trace\'s Nyquist frequency (%s Hz).' % (
                    self.filter.highcut, st[0].stats.sampling_frequency/2))
        try:
            st.remove_response()
        except ValueError:
            # missing station response
            st.sort(keys=['starttime'])
            ninv = self.store_client.rclient.get_stations(
                network=st[0].stats.network, station=st[0].stats.station,
                channel='*', starttime=st[0].stats.starttime,
                endtime=st[-1].stats.endtime, level='response')
            st.attach_response(ninv)
            st.remove_response()
            self.store_client._write_inventory(ninv)
        for tr in st:
            # The actual data in the mseeds was changed from int to float64
            # now,
            # Save some space by changing it back to 32 bit (most of the
            # digitizers work at 24 bit anyways)
            tr.data = np.require(tr.data, np.float32)
        st.detrend()
        st.taper(
            max_percentage=0.05, type='hann', max_length=2, side='both')
        st.filter(
            'bandpass', freqmin=self.filter.lowcut,
            freqmax=self.filter.highcut, zerophase=True)
        # Downsmaple
        st.resample
        try:
            return st, ninv
        except NameError:
            # read inventory
            ninv = self.store_client.read_inventory().select(
                network=tr.stats.network, station=tr.stats.station)
            return st, ninv
