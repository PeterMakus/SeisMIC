'''
Module to handle the different h5 files.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 16th March 2021 04:00:26 pm
Last Modified: Thursday, 18th March 2021 04:25:20 pm
'''

import os

import numpy as np
from obspy import UTCDateTime
from pyasdf import ASDFDataSet

from miic3.plot.plt_spectra import plot_spct_series, spct_series_welch
from miic3.trace_data.waveform import Store_Client
from miic3.trace_data.preprocess import Preprocessor


class NoiseDB(object):
    """
    [summary]
    """

    def __init__(
            self, dir: str, network: str, station: str) -> None:
        super().__init__()
        # Location of the h5 file
        self.loc = os.path.join(dir, '%s.%s.h5' % (network, station))

        with ASDFDataSet(self.loc) as ds:
            # get dict a bit convoluted but see pyasdf documentation
            self.param = ds.auxiliary_data.PreprocessingParameters.\
                param.parameters

        self.filter = self.param["filter"]
        self.sampling_rate = self.param["sampling_rate"]
        self.station = station
        self.network = network

    def return_preprocessor(self, store_client: Store_Client) -> Preprocessor:

        return Preprocessor(
            store_client, self.filter, self.sampling_rate,
            os.path.dirname(self.loc))

    def plot_spectrum(
        self, window_length: int = 4*3600,
        starttime: UTCDateTime or None = None,
        endtime: UTCDateTime or None = None, save_spectrum: bool = True,
        norm: str or None = None, norm_method: str = None, title: str = None,
            outfile: str = None, fmt: str = 'pdf', dpi: str = 300):
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
                l = []
                for item in A.files:
                    l.append(A[item])
                f, t, S = l
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

    def get_active_times(self) -> tuple:
        """
        Returns earliest available Starttime and latest available endtime.

        :return: (UTCDateTime, UTCDateTime)
        :rtype: tuple
        """
        with ASDFDataSet(self.loc) as ds:
            st = ds.waveforms["%s.%s" % (self.network, self.station)].processed
            st.sort(keys=['starttime'])
            starttime = st[0].stats.starttime
            endtime = st[-1].stats.endtime
        return (starttime, endtime)
