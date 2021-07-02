'''
Module to handle the different h5 files.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 16th March 2021 04:00:26 pm
Last Modified: Friday, 2nd July 2021 11:55:56 am
'''

from glob import glob
import os
from typing import Tuple

import numpy as np
from obspy import UTCDateTime, Stream, Inventory
from pyasdf import ASDFDataSet
import pyasdf

from miic3.plot.plt_spectra import plot_spct_series, spct_series_welch
from miic3.trace_data.waveform import Store_Client
# from miic3.trace_data.preprocess import Preprocessor


h5_FMTSTR = os.path.join("{dir}", "{network}.{station}.h5")


class NoDataError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class NoiseDB(object):
    """
    Class to handle ASDF files that contain the preprocessed noise data.
    """

    def __init__(
            self, dir: str, network: str, station: str):
        """
        Initialises the noise database (ASDF handler).

        :param dir: Directory to create the file in
        :type dir: str
        :param network: Network code
        :type network: str
        :param station: Station code
        :type station: str
        """

        assert isinstance(station, str) and isinstance(network, str), \
            "Station and network have to be provided as strings."
        assert '*' not in dir+network+station or '?' not in \
            dir+network+station, "Wildcards are not supported."

        super().__init__()
        # Location of the h5 file
        self.loc = h5_FMTSTR.format(dir=dir, network=network, station=station)

        with ASDFDataSet(self.loc, mode='r', mpi=False) as ds:
            # get dict a bit convoluted but see pyasdf documentation
            self.param = ds.auxiliary_data.PreprocessingParameters.\
                param.parameters

        self.sampling_rate = self.param["sampling_rate"]
        self.station = station
        self.network = network

    def return_preprocessor(self, store_client: Store_Client) -> dict:
        """
        Returns the properties of
        the :class:`~miic3.trace_data.preprocess.Preprocessor` that
        can be used to add data to this NoiseDB.

        :param store_client: The Store Client that should be used to obtain
            the new data.
        :type store_client: Store_Client
        :return: The dictionary that can be used to initialise the correct
            preprocessor.
        :rtype: Preprocessor
        """
        kwargs = {
            'store_client': store_client,
            'sampling_rate': self.sampling_rate,
            'outfolder': os.path.dirname(self.loc),
            'remove_response': self.param["remove_response"],
            '_ex_times': self.get_active_times()}
        return kwargs

    def plot_spectrum(
        self, window_length: int = 4*3600,
        starttime: UTCDateTime or None = None,
        endtime: UTCDateTime or None = None, save_spectrum: bool = True,
        norm: str or None = None, norm_method: str = None, title: str = None,
            outfile: str = None, fmt: str = 'pdf', dpi: int = 300):
        """
        Plots a spectral time series of the data in the requested time window.
        The computation is done by employing welch windows with an arbitrary
        window_length.

        :param window_length: Window length of each spectrum in seconds,
            defaults to 4 hours.
        :type window_length: int, optional
        :param starttime: Starttime. Set to None if you wish to use the
            earliest available time, defaults to None
        :type starttime: UTCDateTimeorNone, optional
        :param endtime: Endtime. Set to None if you wish to use the latest
            available, defaults to None
        :type endtime: UTCDateTimeorNone, optional
        :param save_spectrum: Save the spectrum to a .npz file located in the
            same directory as the .h5 file. Defaults to True
        :type save_spectrum: bool, optional
        :param norm: Normalisation to be used before plotting. Available
            options are `'f'` or `'t'` for normalisation over frequency
            or time axis, respectively, defaults to None
        :type norm: strorNone, optional
        :param norm_method: Method to normalise by. Either `'linalg'` (i.e.,
            length of vector), `'mean'`, or `'median'`, defaults to None.
        :type norm_method: str, optional
        :param title: Title of the plot, defaults to None
        :type title: str, optional
        :param outfile: File to save the figure to, defaults to None
        :type outfile: str, optional
        :param fmt: Format to save the figure in, defaults to 'pdf'
        :type fmt: str, optional
        :param dpi: *For non-vector formats:* DPI to save the plot with
            , defaults to 300.
        :type dpi: str, optional

        .. note:: If a saved spectrum with similar properties is found, the
            computation will be skipped.
        """

        # When processing with open beginning/end is desired.
        if not starttime or not endtime:
            st, et = self.get_active_times()
            starttime = starttime or st
            endtime = endtime or et

        # Assigned name to save the spectral series
        fname = '%s.%s_%s_%s_spectrum.npz' % (
            self.network, self.station, starttime, endtime)
        out_spct = os.path.join(os.path.dirname(self.loc), fname)

        if os.path.exists(out_spct):
            with np.load(out_spct) as A:
                specl = []
                for item in A.files:
                    specl.append(A[item])
                f, t, S = specl
        else:
            with ASDFDataSet(self.loc) as ds:
                for station in ds.ifilter(
                        ds.q.starttime >= starttime, ds.q.endtime <= endtime):
                    st = station.processed
                    f, t, S = spct_series_welch(st, window_length)

            if save_spectrum:
                np.savez(out_spct, f, t, S)

        # plotting
        plot_spct_series(
            S, f, t, title=title, outfile=outfile, fmt=fmt, dpi=dpi, norm=norm,
            norm_method=norm_method)

    def get_active_times(self) -> Tuple[UTCDateTime, UTCDateTime]:
        """
        Returns earliest available Starttime and latest available endtime.

        :return: (UTCDateTime, UTCDateTime)
        :rtype: tuple
        """

        with ASDFDataSet(self.loc, mode="r", mpi=False) as ds:
            try:
                st = []
                for k, v in ds.waveforms["%s.%s" % (
                        self.network, self.station)].get_waveform_attributes(
                            ).items():
                    if 'processed' not in k:
                        continue
                    st.append(v['starttime'])
            except pyasdf.exceptions.WaveformNotInFileException:
                # This happens when there are no times at all
                return (None, None)
        starttime = convert_asdf_time_to_utcdt(min(st))
        endtime = convert_asdf_time_to_utcdt(max(st)) + 24*3600
        return (starttime, endtime)

    def get_all_data(
            self, window_length: int = 3600, increment: int = 3600) -> Stream:
        """
        Returns all available data to a single :class:`~obspy.core.Stream`.

        :param window_length: Length in seconds of each Trace in the Stream.
        :type window_length: int, optional
        :param increment: Step between each of the Windows in seconds.
        :type increment: int, optional
        :return: Stream object containing daily traces.
        :rtype: :class:`~obspy.core.Stream`

        .. warning:: This will fail if the dataset is too large.
        """

        with ASDFDataSet(self.loc, mode="r", mpi=False) as ds:
            st = ds.waveforms["%s.%s" % (self.network, self.station)].processed
        # Make sure all the time windows have an equal length
        st2 = Stream()
        for windowed_st in st.slide(
                window_length=window_length, step=increment,
                include_partial_windows=True):
            st2.extend(windowed_st)
        return st2

    def get_time_window(self, start: UTCDateTime, end: UTCDateTime) -> Stream:
        """
        Return a Stream between the requested times.

        :param start: Starttime
        :type start: UTCDateTime
        :param end: Endtime
        :type end: UTCDateTime
        :raises NoDataError: If the stream to be returned is empty
        :return: Stream containing all data in the requested time frame.
        :rtype: Stream

        .. note:: Will not raise an error if the time is only partly available.
        However, this will raise an error if there is no data at all.
        """
        with ASDFDataSet(self.loc, mode="r", mpi=False) as ds:
            st = ds.get_waveforms(
                self.network, self.station, '*', '*', start, end, 'processed',
                automerge=True)
            st.merge()  # To return one trace rather than several
        if not st.count():
            raise NoDataError('No Data between requested start and end.')
        return st

    def get_inventory(self) -> Inventory:
        """
        Returns the StationXML belonging to the station for that the data
        has been preprocessed.

        :return: Inventory object.
        :rtype: :class:`~obspy.core.inventory.Inventory`
        """
        with ASDFDataSet(self.loc, mode="r", mpi=False) as ds:
            inv = ds.waveforms[
                "%s.%s" % (self.network, self.station)].StationXML
        return inv


def get_available_stations(
        dir: str, network: str, station: str) -> Tuple[list, list]:
    """
    Returns two lists, decribing the stations with available preprocessed data.
    The first list contains the stations' network codes, the second the station
    codes.

    :param dir: directory to search in
    :type dir: str
    :param network: network, file patterns allowed
    :type network: str
    :param station: stations, file patterns allowed
    :type station: str
    :return: two lists `(networkcodes, stationcodes)`
    :rtype: tuple
    """
    pattern = h5_FMTSTR.format(dir=dir, network=network, station=station)
    pathl = glob(os.path.join(dir, pattern))
    if not len(pathl):
        raise FileNotFoundError(
            """No files found for the requested network \
        and station combination.""")
    codea = np.array(
        [path.split(os.path.sep)[-1].split('.') for path in pathl])
    netlist = list(codea[:, 0])
    statlist = list(codea[:, 1])
    return netlist, statlist


def convert_asdf_time_to_utcdt(asdf_time: int) -> UTCDateTime:
    return UTCDateTime(asdf_time*1e-9)
