'''
A module to create seismic ambient noise correlations.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 4th March 2021 03:54:06 pm
Last Modified: Thursday, 6th May 2021 12:06:46 pm
'''
from collections import namedtuple
from glob import glob
import os
from warnings import warn

import numpy as np
from obspy import UTCDateTime, Stream
from obspy.core.inventory.inventory import Inventory
from pyasdf.asdf_data_set import ASDFDataSet

from .waveform import Store_Client
from miic3.db.asdf_handler import NoiseDB
from miic3.utils.miic_utils import resample_or_decimate


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
        self, store_client: Store_Client, sampling_rate: float, outfolder: str,
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
        # assert sampling_rate/2 > filter[1], \
        #     "The highcut frequency of the filter has to be lower than the \
        #         Nyquist frequency (sampling frequency/2) of the signal to \
        #             prevent aliasing. Current Nyquist frequency is: %s."\
        #                  % str(sampling_rate/2)
        self.store_client = store_client
        # Probably importannt to save the filter and perhaps also the
        # sampling frequency in the asdf file, so one can perhaps start a
        # Preprocessor by giving the file as parameter.
        self.sampling_rate = sampling_rate
        self.outloc = os.path.join(store_client.sds_root, outfolder)

        # Create preprocessing parameter dict
        self.param = {
            "sampling_rate": self.sampling_rate,
            "outfolder": self.outloc}

        # Add the already existing times
        self.ex_times = _ex_times

    def preprocess_bulk(
        self, network: str or None = None,
        starttime: UTCDateTime or None = None,
        endtime: UTCDateTime or None = None,
            backend: str = 'joblib', n_cpus: int = -1):
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
        :param backend: Backend to use for the multi-threading. If you are
            starting this script with mpirun, set it `=mpi`. Defaults to
            `joblib` (which uses the loky backend).
        :type backend: str, optional
        :param n_cpus: Only relevant if backend=joblib. Overwrites the number
            of cpu cores that should be used. Joblib tends to oversubscribe
            resources since it uses all virtual cores (i.e., threads), so it
            can be useful to set this equal your number of **physical** cores
            to achieve the best perfomance (expect a gain of about 15%) and
            save some memory. -1 means all available **virtual** CPUs will be
            used. 1 is useful for debugging. Default to -1.
        :type n_cpus: int, optional
        """
        if backend == 'mpi':
            from mpi4py import MPI
            statlist = self._all_stations_raw(network)
            comm = MPI.COMM_WORLD
            psize = comm.Get_size()
            rank = comm.Get_rank()
            # decide which process processes which station
            pmap = (np.arange(len(statlist))*psize)/len(statlist)
            pmap = pmap.astype(np.int32)
            ind = pmap == rank
            ind = np.arange(len(statlist))[ind]
            for ii in ind:
                self._preprocess_mc_init(
                    statlist[ii][0], statlist[ii][1], '*', '*', starttime,
                    endtime)
            comm.barrier()
        elif backend == 'joblib':
            from joblib import Parallel, delayed
            Parallel(n_jobs=n_cpus, verbose=8)(
                        delayed(self._preprocess_mc_init)(
                            network, station, '*', '*', starttime, endtime)
                        for network, station in self._all_stations_raw(
                            network))
        else:
            msg = 'Backend "%s" not supported' % backend
            raise ValueError(msg)

    def _preprocess_mc_init(
        self, network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime or None = None,
            endtime: UTCDateTime or None = None):
        """
        Same as preprocess, but with automatic handlers for the inbuilt
        exceptions.
        """
        try:
            self.preprocess(
                network, station, location, channel, starttime, endtime)
        except PermissionError:
            # Now we get the appropraite preprocessor and do a preprocess on
            # that one
            try:
                ndb = NoiseDB(self.outloc, network, station)
                kwargs = ndb.return_preprocessor(self.store_client)
                p = Preprocessor(**kwargs)
                p.preprocess(
                    network, station, location, channel, starttime, endtime)
            except FileExistsError:
                print('No new data for station %s.%s processed.' % (
                    network, station))

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
            raise PermissionError(msg)

        if not starttime or not endtime:
            warn(
                "Returned start and endtimes will not be checked due to\
             wildcard.", category=UserWarning)
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
            if self.ex_times[0]-5 <= starttime and \
                    self.ex_times[1]+5 >= endtime:
                raise FileExistsError('No new data found. All data have already \
                been preprocessed?')
            # New new data
            elif self.ex_times[0]-5 <= starttime and \
                    self.ex_times[1]+5 < endtime:
                req_start = [self.ex_times[1]]
            # New old data
            elif self.ex_times[0]-5 > starttime and \
                    self.ex_times[1]+5 >= endtime:
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
        # Bit that needs to be added on each side for the extra taper
        delta = 3600*0.05
        for starttime, endtime in zip(req_start, req_end):
            if endtime-starttime > 24*3600+2*delta:
                while starttime < endtime:
                    starttimes.append(starttime-delta)
                    # Skip two seconds, so the same file is not loaded twice
                    starttime = starttime+24*3600
                    endtimes.append(starttime+delta)
            else:
                starttimes.append(starttime)
                endtimes.append(endtime)

        chunk_len = 3600  # processing chunk length in s
        # One hour is pretty close to the optimal length
        # chunk_len = next_fast_len(chunk_len) are identical

        # for starttime, endtime in zip(starttimes, endtimes):
        for starttime, endtime in zip(starttimes, endtimes):
            # Return obspy stream with data from this station if the data
            # does not already exist
            st_raw = self.store_client.get_waveforms(
                network, station, location, channel, starttime, endtime,
                _check_times=_check_times)
            st_raw.sort()

            st_proc = Stream()
            # Devide chunk_len by .9 to compensate for the taper in response
            # removal
            for ii, st in enumerate(st_raw.slide(
                chunk_len/.9, chunk_len,
                    include_partial_windows=True)):
                try:
                    st, resp = self._preprocess(
                        st, self.store_client.inventory)
                    # Cut the tapered parts off again
                    delta = (st[0].stats.endtime-st[0].stats.starttime)*0.05
                    # if st[0].stats.starttime == st_raw[0].stats.starttime:
                    #     st.trim(
                    #         starttime=st[0].stats.starttime,
                    #         endtime=st[0].stats.endtime-2*delta)
                    # elif st[0].stats.endtime == st_raw[-1].stats.endtime:
                    #     continue
                    # else:
                    st.trim(
                        starttime=st[0].stats.starttime+delta,
                        endtime=st[0].stats.endtime-delta)
                    st_proc.extend(st)
                except FrequencyError as e:
                    warn(e + ' Trace is skipped.')
                    continue

            # Create folder if it does not exist
            os.makedirs(self.outloc, exist_ok=True)

            with ASDFDataSet(outfile, mpi=False) as ds:
                ds.add_waveforms(st_proc, tag='processed')

        with ASDFDataSet(outfile, mpi=False) as ds:
            # Save some processing values as auxiliary data
            ds.add_auxiliary_data(
                data=np.empty(1), data_type='PreprocessingParameters',
                path='param', parameters=self.param)
            ds.add_stationxml(resp)

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
            os.path.join(self.store_client.sds_root, '????', network, '*'))
        statlist = []
        for path in oslist:
            if not isinstance(eval(path.split('/')[-3]), int):
                continue
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
        st.sort(keys=['starttime'])
        # Check sampling frequency
        if self.sampling_rate > st[0].stats.sampling_rate:
            raise FrequencyError(
                'The new sample rate (%sHz) is higher than the trace\'s native\
            sample rate (%s Hz).' % (
                    str(self.sampling_rate), str(
                        st[0].stats.sampling_rate)))

        st.filter('lowpass_cheby_2', freq=(self.sampling_rate/2)*.9)
        # Downsample
        st = resample_or_decimate(st, self.sampling_rate)

        if inv:
            ninv = inv
            st.attach_response(ninv)
        try:
            st.remove_response()
        except ValueError:
            print('Station response not found ... loading from remote.')
            # missing station response
            ninv = self.store_client.rclient.get_stations(
                network=st[0].stats.network, station=st[0].stats.station,
                channel='*', starttime=st[0].stats.starttime,
                endtime=st[-1].stats.endtime, level='response')
            st.attach_response(ninv)
            st.remove_response()
            self.store_client._write_inventory(ninv)

        st.detrend()
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


def detrend(data: np.ndarray) -> np.ndarray:
    '''
    From NoisPy (Jian et. al., 2020)


    this function removes the signal trend based on QR decomposion
    NOTE: QR is a lot faster than the least square inversion used by
    scipy (also in obspy).
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with trend removed
    '''
    # I ended up not using it because it seems to be slower after all.
    if data.ndim == 1:
        npts = data.shape[0]
        X = np.ones((npts, 2))
        X[:, 0] = np.arange(0, npts)/npts
        Q, R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R), Q.transpose())
        coeff = np.dot(rq, data)
        data = data-np.dot(X, coeff)
    elif data.ndim == 2:
        npts = data.shape[1]
        X = np.ones((npts, 2))
        X[:, 0] = np.arange(0, npts)/npts
        Q, R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R), Q.transpose())
        for ii in range(data.shape[0]):
            coeff = np.dot(rq, data[ii])
            data[ii] = data[ii] - np.dot(X, coeff)
    return data
