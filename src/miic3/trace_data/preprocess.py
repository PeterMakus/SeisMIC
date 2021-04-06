'''
A module to create seismic ambient noise correlations.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 4th March 2021 03:54:06 pm
Last Modified: Tuesday, 30th March 2021 09:30:25 am
'''
from collections import namedtuple
from glob import glob
import os
from warnings import warn

from joblib import Parallel, delayed, cpu_count
import numpy as np
from obspy import UTCDateTime, Stream
from obspy.core.inventory.inventory import Inventory
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
    """
    Object that manages the preprocessing of raw ambient noise data. The
    preprocessed files will be saved as an ASDF (h5) file.

    """
    def __init__(
        self, store_client: Store_Client, filter: tuple,
        sampling_rate: float, outfolder: str,
            _ex_times: tuple = None) -> None:
        """
        Create the Preprocesser object.

        :param store_client: The associated Store_Client object that finds the
            files in the sds structured database.
        :type store_client: Store_Client
        :param filter: Frequencies to filter with. Provided as tuple in the
            form (lowcut_frequency, highcut_frequency). Note that the highcut
            frequency must be lower or equal to the Nyquist frequency
            (i.e., half of the defined sampling frequency).
        :type filter: tuple
        :param sampling_rate: Frequency that the data will be
            resampled to. All raw data with a lower native sampling frequency
            will be discarded!
        :type sampling_rate: float
        :param outfolder: Name of the folder to write the asdf files to. Note
            that this is only the daughterdir under sds_root. **Must not be
            `inventory` or any year (e.g., `1975`)!**
        :type outfolder: str
        :param _extimes: Already processed times. This is asserted
            automatically by :class:`~miic3.asdf_handler.NoiseDB`.
            Defaults to None.
        :type _extimes: tuple, optional

        .. warning:: Pay close attention to the correct definition of
            filter and sampling frequencies. All raw data with a lower native
            sampling frequency than the one defined will be discarded!
        """
        super().__init__()
        assert sampling_rate/2 > filter[1], \
            "The highcut frequency of the filter has to be lower than the \
                Nyquist frequency (sampling frequency/2) of the signal to \
                    prevent aliasing. Current Nyquist frequency is: %s."\
                         % str(sampling_rate/2)
        self.store_client = store_client
        # Probably importannt to save the filter and perhaps also the
        # sampling frequency in the asdf file, so one can perhaps start a
        # Preprocessor by giving the file as parameter.
        self.filter = Filter(filter[0], filter[1])
        self.sampling_rate = sampling_rate
        self.outloc = os.path.join(store_client.sds_root, outfolder)

        # Create preprocessing parameter dict
        self.param = {
            "filter": self.filter,
            "sampling_rate": self.sampling_rate,
            "outfolder": self.outloc}

        # Add the already existing times
        self.ex_times = _ex_times

    def preprocess_bulk(
        self, network: str or None = None,
        starttime: UTCDateTime or None = None,
            endtime: UTCDateTime or None = None):
        """
        Preprocesses data from several stations in parallel. Writes
        preprocessed files to asdf file (one per station).

        :param network: Network code, wildcards allowed. Use `None` if you wish
            to process all available data, defaults to None.
        :type network: strorNone, optional
        :param starttime: Starttime. Use `None` to process from the earliest
            time, defaults to None
        :type starttime: UTCDateTimeorNone, optional
        :param endtime: Endtime. If all available times should be used,
            set `=None`. Defaults to None, defaults to None.
        :type endtime: UTCDateTimeorNone, optional
        """
        # 16.03.21 This is awfully slow. However the single processes are not
        # Maybe playing with chunk size to prevent overheating?
        Parallel(n_jobs=-1)(
                    delayed(self.preprocess)(
                        network, station, '*', '*', starttime, endtime)
                    for network, station in self._all_stations_raw(network))

    def preprocess(
        self, network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime or None = None,
            endtime: UTCDateTime or None = None):
        """
        Function to preprocess data from one single station. Writes
        preprocessed streams to asdf file.

        :param network: Network code.
        :type network: str
        :param station: Station Code.
        :type station: str
        :param location: Location Code, wildcards allowed.
        :type location: str
        :param channel: Channel Code, wildcards allowed.
        :type channel: str
        :param starttime: Starttime. If all available times should be used,
            set = None. Defaults to None
        :type starttime: UTCDateTimeorNone, optional
        :param endtime: Endtime. If all available times should be used,
            set = None. Defaults to None, defaults to None
        :type endtime: UTCDateTimeorNone, optional

        .. seealso:: For processing of several stations or network use
                    :func:`~Preprocessor.preprocess_bulk()`.
        """

        assert '*' not in station or type(station) != list, 'To process data\
             from several stations, use \
                 :func:`~Preprocessor.preprocess_bulk()`!'

        # Check whether there is already an existing output file and if
        # make sure that the processing parameters are the same!
        outfile = os.path.join(self.outloc, '%s.%s.h5' % (network, station))

        if os.path.exists(outfile) and not self.ex_times:
            msg = "The output file already exists. Use \
                :func:`~miic3.db.asdf_handler.NoiseDB.return_preprocessor()` \
                    to yield the correct Preprocessor!"
            raise ValueError(msg)

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

        # check already existing times, there is one case with two request
        # chunks, so we will have to create a list for times called
        # req_start and req_end
        # standard way with only new times:
        req_start = [starttime]
        req_end = [endtime]
        if self.ex_times:
            # No new data at all
            if self.ex_times[0] <= starttime and self.ex_times[1] >= endtime:
                raise ValueError('No new data found. All data has already \
                    been preprocessed?')
            # New new data
            elif self.ex_times[0] <= starttime and self.ex_times[1] < endtime:
                req_start = [self.ex_times[1]]
            # New old data
            elif self.ex_times[0] > starttime and self.ex_times[1] >= endtime:
                req_end = [self.ex_times[0]]
            # New data on both sides
            else:
                req_end = [self.ex_times[0], endtime]
                req_start = [starttime, self.ex_times[1]]

        # Save some memory: will cut the requests in shorter chunks.
        # Else this becomes problematic especially, when processing with
        # several threads at the same time and the very long streams have to be
        # held in RAM.
        starttimes = []
        endtimes = []
        for starttime, endtime in zip(req_start, req_end):
            if endtime-starttime > 3600*6:
                while starttime < endtime:
                    starttimes.append(starttime)
                    starttime = starttime+3600*6
                    endtimes.append(starttime)
            else:
                starttimes.append(starttime)
                endtimes.append(endtime)

        for starttime, endtime in zip(starttimes, endtimes):
            # Return obspy stream with data from this station if the data
            # does not already exist
            st = self.store_client.get_waveforms(
                network, station, location, channel, starttime, endtime,
                _check_times=_check_times)

            try:
                st, resp = self._preprocess(st)
            except FrequencyError as e:
                warn(e + ' Trace is skipped.')
                continue

            # Create folder if it does not exist
            os.makedirs(self.outloc, exist_ok=True)

            with ASDFDataSet(outfile) as ds:
                ds.add_waveforms(st, tag='processed')
                ds.add_stationxml(resp)

        with ASDFDataSet(outfile) as ds:
            # Save some processing values as auxiliary data
            ds.add_auxiliary_data(
                data=np.empty(1), data_type='PreprocessingParameters',
                path='param', parameters=self.param)

    def _all_stations_raw(
            self, network: str or None = None) -> list:
        """
        Returns a list of stations for which raw data is available.
        Very similar to the
        :func:`~obspy.clients.filesystem.sds.get_all_stations()` method with
        the difference that this one allows to filter for particular networks.

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

    def _preprocess(self, st: Stream, inv: Inventory or None = None) -> Stream:
        """
        Private method that executes the preprocessing steps on a *per Stream*
        basis.
        """
        # Check sampling frequency
        if self.sampling_rate > st[0].stats.sampling_rate:
            raise FrequencyError(
                'The new sample rate (%sHz) is higher than the trace\'s native\
                     sample rate (%s Hz).' % (
                    str(self.filter.highcut), str(
                        st[0].stats.sampling_rate/2)))
        if inv:
            ninv = inv
            st.attach_response(ninv)
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

        st.detrend()
        # st.taper(
        #     max_percentage=0.05, type='hann', max_length=2, side='both')
        # At this point, it probably does not make too much sense to taper.
        st.filter(
            'bandpass', freqmin=self.filter.lowcut,
            freqmax=self.filter.highcut, zerophase=True)
        # Downsample
        st.resample(self.sampling_rate)

        for tr in st:
            # !Last operation before saving!
            # The actual data in the mseeds was changed from int to float64
            # now,
            # Save some space by changing it back to 32 bit (most of the
            # digitizers work at 24 bit anyways)
            tr.data = np.require(tr.data, np.float32)
        try:
            return st, ninv
        except (UnboundLocalError, NameError):
            # read inventory
            ninv = self.store_client.read_inventory().select(
                network=tr.stats.network, station=tr.stats.station)
            return st, ninv
