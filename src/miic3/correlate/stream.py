'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th April 2021 04:19:35 pm
Last Modified: Monday, 19th July 2021 03:55:19 pm
'''
from typing import Iterator, List, Tuple
from copy import deepcopy
import warnings
from matplotlib import pyplot as plt


import numpy as np
from obspy import Stream, Trace, Inventory, UTCDateTime
from obspy.core import Stats

from miic3.utils import miic_utils as m3ut
from miic3.plot.plot_correlation import plot_cst, plot_ctr
import miic3.monitor.post_corr_process as pcp
from miic3.monitor.stretch_mod import time_stretch_apply
from miic3.monitor.dv import DV
from miic3.correlate.stats import CorrStats


class CorrBulk(object):
    """
    An object for faster computations on several correlations. The input
    correlation contain data from only one Station-Channel pair.
    """
    def __init__(
        self, A: np.ndarray, stats: CorrStats = None,
            statlist: List[CorrStats] = None):
        self.data = A
        if stats:
            self.stats = stats
        elif statlist:
            self.stats = convert_statlist_to_bulk_stats(statlist)
        else:
            self.stats = CorrStats()
            self.stats['ntrcs'], self.stats['npts'] = A.shape
        self.stats['processing_bulk'] = []
        self.ref_trc = None

    def normalize(
        self, starttime: float = None, endtime: float = None,
            normtype: str = 'energy'):
        """
        Correct amplitude variations with time.

        Measure the maximum of the absolute value of the correlation matrix in
        a specified lapse time window and normalize the correlation traces by
        this values. A coherent phase in the respective lapse time window will
        have constant ampitude afterwards..

        :type starttime: float
        :param starttime: Beginning of time window in seconds with respect to
            the zero position.
        :type endtime: float
        :param endtime: end time window in seconds with respect to the zero
            position.
        :type normtype: string
        :param normtype: one of the following 'energy', 'max', 'absmax',
            'abssum' to decide about the way to calculate the normalization.

        :rtype: CorrBulk
        :return: Same object as in self, but normalised.

        ..note:: This action is performed **in-place**. If you would like to
            keep the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`.
        """
        self.data = pcp.corr_mat_normalize(
            self.data, self.stats, starttime, endtime, normtype)
        proc_str = 'normalize; normtype: %s' % normtype
        if starttime and endtime:
            proc_str += ', starttime: %s, endtime: %s' % (
                str(starttime), str(endtime))
        self.stats.processing_bulk += [proc_str]
        return self

    def copy(self):
        """
        Returns a copy of self

        :return: A copy of self
        """
        return deepcopy(self)

    def correct_decay(self):
        """
        Correct for the amplitude decay in a correlation matrix.

        Due to attenuation and geometrical spreading the amplitude of the
        correlations decays with increasing lapse time. This decay is corrected
        by dividing the correlation functions by an exponential function that
        models the decay.

        :return: Self but with data corrected for amplitude decay

        ..note:: This action is performed **in-place**. If you would like to
            keep the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`.
        """
        self.data = pcp.corr_mat_correct_decay(self.data, self.stats)
        self.stats.processing_bulk += ['Corrected for Amplitude Decay']
        return self

    def correct_stretch(self, dv: DV):
        """
        Correct stretching of correlation matrix

        In the case of a homogeneous subsurface velocity change the correlation
        traces are stretched or compressed. This stretching can be measured
        with `self.stretch`. The resulting `DV` object can be passed to
        this function to remove the stretching from the correlation matrix.

        :param dv: Velocity Change object
        :type dv: DV

        ..note:: This action is performed **in-place**. If you would like to
            keep the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`.
        """
        self.data = time_stretch_apply(self.data, -1.*dv.value)
        self.stats.processing_bulk += ['Applied time stretch']
        return self

    def envelope(self):
        """
        Calculate the envelope of a correlation matrix.

        The corrlation data of the correlation matrix are replaced by their
        Hilbert envelopes.


        :return: self with the envelope in data

        ..note:: This action is performed **in-place**. If you would like to
            keep the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`.
        """
        self.data = pcp.corr_mat_envelope(self.data)
        self.stats.processing_bulk += ['Computed Envelope']
        return self

    def filter(self, freqs: Tuple[float, float], order: int = 3):
        """
        Filters the correlation matrix in the frequency band specified in
        freqs using a zero phase filter of twice the order given in order.

        :type freqs: Tuple
        :param freqs: lower and upper limits of the pass band in Hertz
        :type order: int
        :param order: half the order of the Butterworth filter

        :return: self

        ..note:: This action is performed **in-place**. If you would like to
            keep the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`.
        """
        self.data = pcp.corr_mat_filter(self.data, self.stats, freqs, order)
        proc = ['filter; freqs: %s, order: %s' % (str(freqs), str(order))]
        self.stats.processing_bulk += proc
        return self

    def extract_trace(
        self, method: str = 'mean',
            percentile: float = 50.) -> np.ndarray:
        """
        Extract a representative trace from a correlation matrix.

        Extract a correlation trace from the that best represents the
        correlation matrix. ``Method`` decides about method to extract the
        trace. The following possibilities are available

        * ``mean`` averages all traces in the matrix
        * ``norm_mean`` averages the traces normalized after normalizing for
            maxima
        * ``similarity_percentile`` averages the ``percentile`` % of traces
            that best correlate with the mean of all traces. This will exclude
            abnormal traces. ``percentile`` = 50 will return an average of
            traces with correlation (with mean trace) above the median.

        :type method: string
        :param method: method to extract the trace
        :type percentile: float
        :param percentile: only used for method=='similarity_percentile'

        :rtype: np.ndarray
        :return: extracted trace

        ..note:: the extracted trace will also be saved in self.ref_trc
        """
        outdata = pcp.corr_mat_extract_trace(
            self.data, self.stats, method, percentile)
        self.ref_trc = outdata
        return outdata

    def mirror(self):
        """
        Average the causal and acausal (i.e., right and left) parts of the
        correlation.

        ..note:: This action is performed **in-place**. If you would like to
            keep the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`.
        """
        self.data, self.stats = pcp.corr_mat_mirror(self.data, self.stats)
        self.stats.processing_bulk += ['Mirrored.']
        return self

    def resample(
        self, starttimes: List[UTCDateTime],
            endtimes: List[UTCDateTime] = []):
        """ Function to create correlation matrices with constant sampling

        When created from a CorrStream the correlation matrix contains all
        available correlation traces but homogeneous sampling is not guaranteed
        as correlation functions may be missing due to data gaps. This function
        restructures the correlation matrix by inserting or averaging
        correlation functions to provide temporally homogeneous sampling.
        Inserted correlation functions consist of 'nan' if gaps are present and
        averaging is done if more than one correlation function falls in a bin
        between start_times[i] and end_times[i]. If end_time is an empty list
        (default) end_times[i] is set to
        start_times[i] + (start_times[1] - start_times[0])

        :type start_times: list of class:`~obspy.core.UTCDateTime` objects
        :param start_times: list of starting times for the bins of the new
            sampling
        :type end_times: list of class:`~obspy.core.UTCDateTime` objects
        :param end_times: list of end times for the bins of the new
            sampling

        ..note:: This action is performed **in-place**. If you want to keep
            the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`
        """
        self.data, self.stats = pcp.corr_mat_resample(
            self.data, self.stats, starttimes, endtimes)
        self.stats.processing_bulk += [
            'Resampled. Starttimes: %s, Endtimes %s' % (
                str(starttimes), str(endtimes))]
        return self

    def resample_time_axis(self, freq: float):
        """
        Resample the lapse time axis of a correlation matrix. The correlations
        are automatically filtered with a highpass filter of 0.4*sampling
        frequency to avoid aliasing. The function decides automatically whether
        to decimate or resample depending on the desired frequency

        :param freq: new sampling frequency
        :type freq: float

        ..note:: This action is performed **in-place**. If you want to keep
            the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`
        """
        self.data, self.stats = pcp.corr_mat_resample_or_decimate(
            self.data, self.stats, freq)
        self.stats.processing_bulk += [
            'Resampled time axis. New sampling rate: %s' % freq]
        return self

    def smooth(
        self, wsize: int, wtype: str = 'flat',
            axis: int = 1) -> np.ndarray:
        """
        Smoothes the correlation matrix with a given window function of the
        given width along the given axis. This method is based on the
        convolution of a scaled window with the signal. Each row/col
        (i.e. depending on the selected ``axis``) is "prepared" by introducing
        reflected copies of it (with the window size) in both ends so that
        transient parts are minimized in the beginning and end part of the
        resulting array.

        :type wsize: int
        :param wsize: Window size
        :type wtype: string
        :param wtype: Window type. It can be one of:
            ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'] defaults to
            'flat'
        :type axis: int
        :param axis: Axis along with apply the filter. O: smooth along
            correlation lag time axis 1: smooth along time axis

        :rtype: :class:`~numpy.ndarray`
        :return: Filtered matrix

        ..note:: This action is performed **in-place**. If you want to keep
            the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`
        """
        self.data = pcp.corr_mat_smooth(self.data, wsize, wtype, axis)
        self.stats.processing_bulk += [
            'smooth. wsize: %s, wtype: %s, axis: %s'
            % (str(wsize), wtype, str(axis))]
        return self

    def stretch(
        self, ref_trc: np.ndarray = None, tw: List[np.ndarray] = None,
        stretch_range: float = 0.1, stretch_steps: int = 100,
            sides: str = 'both', return_sim_mat: bool = False) -> DV:
        if ref_trc is None:
            ref_trc = self.ref_trc
        dv_dict = pcp.corr_mat_stretch(
            self.data, self.stats, ref_trc, tw, stretch_range, stretch_steps,
            sides, return_sim_mat)
        return DV(**dv_dict)

    def save(self, path: str):
        """
        Save the object to a numpy binary format (**.npz**)

        :param path: Output path
        :type path: str
        """
        kwargs = m3ut.save_header_to_np_array(self.stats)
        np.savez_compressed(
            path, data=self.data, **kwargs)

    def taper(self, width: float):
        """
        Taper the data.

        :param width: width to be tapered in seconds (per side)
        :type width: float

        ..note:: This action is performed **in-place**. If you want to keep
            the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`
        """
        self.data = pcp.corr_mat_taper(self.data, self.stats, width)
        proc = ['tapered: width=%ss' % width]
        self.stats.processing_bulk += proc

    def taper_center(self, width: float, slope_frac: float = 0.05):
        """
        Taper the central part of a correlation matrix.

        Due to electromagnetic cross-talk, signal processing or other effects
        the correlaton matrices are often contaminated around the zero lag
        time. This function tapers (multiples by zero) the central part of
        width `width`. To avoid problems with interpolation and filtering later
        on this is done with cosine taper.

        :param width: width of the central window to be tapered in seconds
            (**total length i.e., not per side**).
        :type width: float
        :param slope_frac: fraction of `width` used for soothing of edges,
            defaults to 0.05
        :type slope_frac: float, optional
        :return: The tapered matrix
        :rtype: np.ndarray

        ..note:: This action is performed **in-place**. If you want to keep
            the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`
        """
        self.data = pcp.corr_mat_taper_center(
            self.data, self.stats, width, slope_frac=slope_frac)
        proc = [
            'tapered-centre: width=%ss, slope_frac=%s' % (width, slope_frac)]
        self.stats.processing_bulk += proc

    def trim(self, starttime: float, endtime: float):
        """

        Trim the correlation matrix to the period from `starttime` to
        `endtime` given in seconds from the zero position, so both can be
        positive and negative.

        :type starttime: float
        :param starttime: start time in seconds with respect to the zero
            position
        :type endtime: float
        :param order: end time in seconds with respect to the zero position

        :return: self

        ..note:: This action is performed **in-place**. If you want to keep
            the original data use
            :func:`~miic3.correlate.stream.CorrelationBulk.copy()`
        """
        self.data, self.stats = pcp.corr_mat_trim(
            self.data, self.stats, starttime, endtime)
        proc = ['trim: %s, %s' % (str(starttime), str(endtime))]
        self.stats.processing_bulk += proc
        return self


def read_corr_bulk(path: str) -> CorrBulk:
    """
    Reads a CorrBulk object from an **.npz** file.

    :param path: Path to file
    :type path: str
    :return: the corresponding and converted CorrBulk object
    :rtype: CorrBulk
    """
    loaded = np.load(path)
    stats = m3ut.load_header_from_np_array(loaded)
    return CorrBulk(loaded['data'], stats=stats)


class CorrStream(Stream):
    """
    Baseclass to hold correlation traces. Basically just a list of the
    correlation traces.
    """
    def __init__(self, traces: list = None):
        self.traces = []
        if isinstance(traces, CorrTrace):
            traces = [traces]
        if traces:
            for tr in traces:
                if not isinstance(tr, CorrTrace):
                    raise TypeError('Traces have to be of type \
                        :class:`~miic3.correlate.correlate.CorrTrace`.')
                self.traces.append(tr)

    def __str__(self, extended=False) -> str:
        """
        Return short summary string of the current stream.

        It will contain the number of Traces in the Stream and the return value
        of each Trace's :meth:`~obspy.core.trace.Trace.__str__` method.

        :type extended: bool, optional
        :param extended: This method will show only 20 traces by default.
            Enable this option to show all entries.

        .. rubric:: Example

        >>> stream = Stream([Trace(), Trace()])
        >>> print(stream)  # doctest: +ELLIPSIS
        2 Trace(s) in Stream:
        ...
        """
        # get longest id
        if self.traces:
            id_length = self and max(len(tr.id) for tr in self) or 0
        else:
            id_length = 0
        out = str(len(self.traces)) + ' Correlation(s) in Stream:\n'
        if len(self.traces) <= 20 or extended is True:
            out = out + "\n".join([_i.__str__(id_length) for _i in self])
        else:
            out = out + "\n" + self.traces[0].__str__() + "\n" + \
                '...\n(%i other correlations)\n...\n' % (len(self.traces) - 2)\
                + self.traces[-1].__str__() + '\n\n[Use "print(' + \
                'Stream.__str__(extended=True))" to print all correlaitons]'
        return out

    def stack(
        self, weight: str = 'by_length', starttime: UTCDateTime = None,
        endtime: UTCDateTime = None, stack_len: int or str = 0,
            regard_location=True):
        """
        Average the data of all traces in the given time windows.
        Will only stack data from the same network/channel/station combination.
        Location codes will only optionally be regarded.

        :param starttime: starttime of the stacking time windows. If None, the
            earliest available is chosen, defaults to None.
        :type starttime: UTCDateTime, optional
        :param endtime: endtime of the stacking time windows. If None, the
            latest available is chosen, defaults to None
        :type endtime: UTCDateTime, optional
        :param stack_len: Length of one stack. Is either a value in seconds,
            the special option "daily" (creates 24h stacks that always start at
            midnight), or 0 for a single stack over the whole time period,
            defaults to 0.
        :type stack_len: intorstr, optional
        :param regard_location: Don't stack correlations with varying location
            code combinations, defaults to True.
        :type regard_location: bool, optional
        :return: A stream holding the stacks.
        :rtype: :class`~miic3.correlate.stream.CorrStream`
        """

        # Seperate if there are different stations channel and or locations
        # involved
        if stack_len == 0:
            return stack_st_by_group(self, regard_location, weight)

        # else
        self.sort(keys=['corr_start'])
        if not starttime:
            starttime = self[0].stats.corr_start
        if not endtime:
            endtime = self[-1].stats.corr_end
        outst = CorrStream()
        if stack_len == 'daily':
            starttime = UTCDateTime(
                year=starttime.year, julday=starttime.julday)
            stack_len = 3600*24
        for st in self.slide(
                stack_len, stack_len, include_partially_selected=True,
                starttime=starttime, endtime=endtime):
            outst.extend(stack_st_by_group(st, regard_location, weight))
        return outst

    def slide(
        self, window_length: float, step: float,
        include_partially_selected: bool = True,
            starttime: UTCDateTime = None, endtime: UTCDateTime = None):
        """
        Generator yielding correlations that are inside of each requested time
        window and inside of this stream.

        Please keep in mind that it only returns a new view of the original
        data. Any modifications are applied to the original data as well. If
        you don't want this you have to create a copy of the yielded
        windows. Also be aware that if you modify the original data and you
        have overlapping windows, all following windows are affected as well.

        Not all yielded windows must have the same number of traces. The
        algorithm will determine the maximal temporal extents by analysing
        all Traces and then creates windows based on these times.


        :param window_length: The length of the requested time window in
            seconds. Note that the window length has to correspond at least to
            the length of the longest correlation window (i.e., the length of
            the correlated waveforms). This is because the correlations cannot
            be sliced.
        :type window_length: float
        :param step: The step between the start times of two successive
            windows in seconds. Has to be greater than 0
        :type step: float
        :param include_partially_selected: If set to ``True``, also the half
            selected time window **before** the requested time will be attached
            Given the following stream containing 6 correlations, "|" are the
            correlation starts and ends, "A" is the requested starttime and "B"
            the corresponding endtime::

                |         |A        |         |       B |         |
                1         2         3         4         5         6

            ``include_partially_selected=True`` will select samples 2-4,
            ``include_partially_selected=False`` will select samples 3-4 only.
            Defaults to True.
        :type include_partially_selected: bool, optional
        :param starttime: Start the sequence at this time instead of the
            earliest available starttime.
        :type starttime: UTCDateTime
        :param endtime: Start the sequence at this time instead of the
            latest available endtime.
        :type endtime: UTCDateTime
        """
        if not starttime:
            starttime = min(tr.stats.corr_start for tr in self)
        if not endtime:
            endtime = max(tr.stats.corr_end for tr in self)

        if window_length < max(
                tr.stats.corr_end-tr.stats.corr_start for tr in self):
            raise ValueError(
                'The length of the requested time window has ' +
                'to be larger or equal than the actual correlation length of' +
                ' one window. i.e., correlations can not be sliced, only ' +
                'selected.')

        if step <= 0:
            raise ValueError('Step has to be larger than 0.')

        windows = np.arange(
            starttime.timestamp, endtime.timestamp, step)

        if len(windows) < 1:
            return

        for start in windows:
            start = UTCDateTime(start)
            stop = start + window_length
            temp = self.select_corr_time(
                start, stop,
                include_partially_selected=include_partially_selected)
            # It might happen that there is a time frame where there are no
            # windows, e.g. two traces separated by a large gap.
            if not temp:
                continue
            yield temp

    def select_corr_time(
        self, starttime: UTCDateTime, endtime: UTCDateTime,
            include_partially_selected: bool = True):
        """
        Selects correlations that are inside of the requested time window.

        :param starttime: Requested start
        :type starttime: UTCDateTime
        :param endtime: Requested end
        :type endtime: UTCDateTime
        :param include_partially_selected: If set to ``True``, also the half
            selected time window **before** the requested time will be attached
            Given the following stream containing 6 correlations, "|" are the
            correlation starts and ends, "A" is the requested starttime and "B"
            the corresponding endtime::

                |         |A        |         |       B |         |
                1         2         3         4         5         6

            ``include_partially_selected=True`` will select samples 2-4,
            ``include_partially_selected=False`` will select samples 3-4 only.
            Defaults to True
        :type include_partially_selected: bool, optional
        :return: Correlation Stream holding all selected traces
        :rtype: CorrStream
        """
        self.sort(keys=['corr_start'])
        outst = CorrStream()
        # the 2 seconds difference are to avoid accidental smoothing
        if include_partially_selected:
            for tr in self:
                if (tr.stats.corr_end > starttime and
                    tr.stats.corr_end < endtime) or \
                        tr.stats.corr_end == endtime:
                    outst.append(tr)
            return outst
        # else
        for tr in self:
            if tr.stats.corr_start >= starttime and\
                 tr.stats.corr_end <= endtime:
                outst.append(tr)
        return outst

    def create_corr_bulk(
        self, network: str = None, station: str = None, channel: str = None,
        location: str = None,
        times: Tuple[UTCDateTime, UTCDateTime] = None,
            inplace=True) -> CorrBulk:
        """
        Creates a CorrelationBulk object, which offers additional options for
        faster postprocessing.

        :param network: Select only this network, defaults to None
        :type network: str, optional
        :param station: Take data from this station, defaults to None
        :type station: str, optional
        :param channel: Take data from this channel, defaults to None
        :type channel: str, optional
        :param location: Take data from only this location. Else various
            locations can be processed together, defaults to None
        :type location: str, optional
        :param times: Only take data from this time window, defaults to None
        :type times: Tuple[UTCDateTime, UTCDateTime], optional
        :param inplace: The original data will be deleted to save memory,
            defaults to True.
        :type inplace: bool, optional
        :return: The CorrelationBulk object
        :rtype: CorrBulk

        .. note:: This function will check whether the metadata of the input
            stream is identical, so that correlations from different stations,
            components, or differently processed data cannot be mixed.
        """
        st = self.select(network, station, location, channel)
        if times:
            st = st.select_corr_time(times[0], times[1])
        A = np.empty((st.count(), st[0].stats.npts))
        statlist = []
        for ii, tr in enumerate(st):
            A[ii] = tr.data
            if inplace:
                del tr.data
            statlist.append(tr.stats)

        stats = convert_statlist_to_bulk_stats(statlist)
        return CorrBulk(A, stats)

    def plot(
        self, sort_by: str = 'corr_start',
        timelimits: Tuple[float, float] = None,
        ylimits: Tuple[float, float] = None, scalingfactor: float = None,
        ax: plt.Axes = None, linewidth: float = 0.25,
            outputfile: str = None, title: str = None):
        """
        Creates a section plot of all correlations in this stream.

        :param sort_by: Which parameter to plot against. Can be either
            ``corr_start`` or ``distance``, defaults to 'corr_start'.
        :type sort_by: str, optional
        :param timelimits: xlimits (lag time) in seconds, defaults to None
        :type timelimits: Tuple[float, float], optional
        :param ylimits: limits for Y-axis (either a :class:`datetime.datetime`
            or float in km (if plotted against distance)), defaults to None.
        :type ylimits: Tuple[float, float], optional
        :param scalingfactor: Which factor to scale the Correlations with. Play
            around with this if you want to make amplitudes bigger or smaller,
            defaults to None (automatically chosen).
        :type scalingfactor: float, optional
        :param ax: Plot in existing axes? Defaults to None
        :type ax: plt.Axes, optional
        :param linewidth: Width of the lines to plot, defaults to 0.25
        :type linewidth: float, optional
        :param outputfile: Save the plot? defaults to None
        :type outputfile: str, optional
        :param title: Title of the plot, defaults to None
        :type title: str, optional

        .. note:: If you would like to plot a subset of this stream, use
            :func:`~miic3.correlate.stream.CorrStream.select`.
        """
        plot_cst(
            self, sort_by=sort_by, timelimits=timelimits, ylimits=ylimits,
            scalingfactor=scalingfactor, ax=ax, linewidth=linewidth,
            outputfile=outputfile, title=title)

    def _to_matrix(
        self, network: str = None, station: str = None, channel: str = None,
        location: str = None,
        times: Tuple[UTCDateTime, UTCDateTime] = None) -> Tuple[
            np.ndarray, Iterator[Stats]]:
        """
        Creates a numpy array from the data in the
        :class:`~miic3.correlate.stream.Stream` object. Also returns a list of
        the Stats objects. The positional arguments are filter arguments.

        :param st: Input Stream
        :type st: CorrStream
        :return: both a numpy array (i.e., matrix) and a list of the stats
        :rtype: Tuple[np.ndarray, Iterator[Stats]]
        """
        st = self.select(network, station, location, channel)
        if times:
            st = st.select_corr_time(times[0], times[1])

        st.sort(keys=['corr_start'])
        A = np.empty((st.count(), st[0].stats.npts))
        stats = []
        for ii, tr in enumerate(st):
            A[ii] = tr.data
            stats.append(tr.stats)
        return A, stats


class CorrTrace(Trace):
    """
    Baseclass to hold correlation data. Derived from the class
    :class:`~obspy.core.trace.Trace`.
    """
    def __init__(
        self, data: np.ndarray, header1: Stats = None,
        header2: Stats = None, inv: Inventory = None,
        start_lag: float = None, end_lag: float = None,
            _header: dict = None):
        """
        Initialise the correlation trace. Is done by combining the stats of the
        two :class:`~obspy.core.trace.Trace` objects' headers. If said headers
        do not contain Station information (i.e., coordinates), an
        :class:`~obspy.core.inventory.Inventory` with information about both
        stations should be provided as well.

        :param data: The correlation data
        :type data: np.ndarray
        :param header1: header of the first trace, defaults to None
        :type header1: Stats, optional
        :param header2: header of the second trace, defaults to None
        :type header2: Stats, optional
        :param inv: Inventory object for the stations, defaults to None
        :type inv: Inventory, optional
        :param start_lag: The lag of the first sample of the correlation given
            in seconds.
        :type start_lag: float
        :param end_lag: The lag of the last sample of the correlation
            in seconds.
        :type end_lag: float
        :param _header: Already combined header, used when reading correlations
            from a file, defaults to None
        :type _header: dict, optional
        """
        if _header:
            header = CorrStats(_header)
        elif not header1 and not header2:
            header = CorrStats()
            if start_lag:
                header['start_lag'] = start_lag
        else:
            header, data = alphabetical_correlation(
                header1, header2, start_lag, end_lag, data, inv)

        super(CorrTrace, self).__init__(data=data)
        self.stats = header
        self.stats['npts'] = len(data)

    def __str__(self, id_length: int = None) -> str:
        """
        Return short summary string of the current trace.

        :rtype: str
        :return: Short summary string of the current trace containing the SEED
            identifier, start time, end time, sampling rate and number of
            points of the current trace.

        .. rubric:: Example

        >>> tr = Trace(header={'station':'FUR', 'network':'GR'})
        >>> str(tr)  # doctest: +ELLIPSIS
        'GR.FUR.. | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples'
        """
        # set fixed id width
        if id_length:
            out = "%%-%ds" % (id_length)
            trace_id = out % self.id
        else:
            trace_id = "%s" % self.id
        out = ''
        # output depending on delta or sampling rate bigger than one
        if self.stats.sampling_rate < 0.1:
            if hasattr(self.stats, 'preview') and self.stats.preview:
                out = out + ' | '\
                    "%(corr_start)s - %(corr_end)s | " + \
                    "%(delta).1f s, %(npts)d samples [preview]"
            else:
                out = out + ' | '\
                    "%(corr_start)s - %(corr_end)s | " + \
                    "%(delta).1f s, %(npts)d samples"
        else:
            if hasattr(self.stats, 'preview') and self.stats.preview:
                out = out + ' | '\
                    "%(corr_start)s - %(corr_end)s | " + \
                    "%(sampling_rate).1f Hz, %(npts)d samples [preview]"
            else:
                out = out + ' | '\
                    "%(corr_start)s - %(corr_end)s | " + \
                    "%(sampling_rate).1f Hz, %(npts)d samples"
        # check for masked array
        if np.ma.count_masked(self.data):
            out += ' (masked)'
        return trace_id + out % (self.stats)

    def plot(
        self, tlim: Tuple[float, float] = None, ax: plt.Axes = None,
            outputdir: str = None, clean: bool = False):
        """
        Plots thios CorrelationTrace.

        :param tlim: Limits for the lapse axis in seconds, defaults to None
        :type tlim: Tuple[float, float], optional
        :param ax: Plot in existing axes, defaults to None
        :type ax: plt.Axes, optional
        :param outputdir: Save this plot? Defaults to None
        :type outputdir: str, optional
        :param clean: Make a clean plot without labels & axes,
            defaults to False.
        :type clean: bool, optional
        """
        plot_ctr(self, tlim, ax, outputdir, clean)

    def times(self) -> np.ndarray:
        """
        Convenience Function that returns an array holding the lag times of the
        correlation.

        :return: Array with lag times
        :rtype: np.ndarray
        """
        return np.arange(
            self.stats.start_lag, self.stats.end_lag + self.stats.delta,
            self.stats.delta)


def alphabetical_correlation(
    header1: Stats, header2: Stats, start_lag: float, end_lag: float,
        data: np.ndarray, inv: Inventory) -> Tuple[CorrStats, np.ndarray]:
    """
    Make sure that Correlations are always created in alphabetical order,
    so that we won't have both a correlation for AB-CD and CD-AB.
    If the correlation was computed in the wrong order, the corr-data will be
    flipped along the t-axis.

    :param header1: Header of the first trace.
    :type header1: Stats
    :param header2: Header of the second trace
    :type header2: Stats
    :param start_lag: start lag in s
    :type start_lag: float
    :param end_lag: end lag in s
    :type end_lag: float
    :param data: The computed cross-correlation for header1-header2
    :type data: np.ndarray
    :param inv: The inventory holding the station coordinates. Only needed if
        coords aren't provided in stats.
    :type inv: Inventory
    :return: the header for the CorrTrace and the data
        (will also be modified in place)
    :rtype: Tuple[Stats, np.ndarray]
    """
    # make sure the order is correct
    # Will do that always alphabetically sorted
    sort1 = header1.network + header1.station + header1.channel
    sort2 = header2.network + header2.station + header2.channel
    sort = [sort1, sort2]
    sorted = sort.copy()
    sorted.sort()
    if sort != sorted:
        header = combine_stats(
            header2, header1, -end_lag,
            inv=inv)
        # reverse array and lag times
        data = np.flip(data)
    else:
        header = combine_stats(
            header1, header2, start_lag,
            inv=inv)
    return header, data


def combine_stats(
    stats1: Stats, stats2: Stats, start_lag: float,
        inv: Inventory = None) -> CorrStats:
    """ Combine the meta-information of two ObsPy Trace.Stats objects

    This function returns a ObsPy :class:`~obspy.core.trace.Stats` object
    obtained combining the two associated with the input Traces.
    Namely ``stats1`` and ``stats2``.

    The fields ['network','station','location','channel'] are combined in
    a ``-`` separated fashion to create a "pseudo" SEED like ``id``.

    For all the others fields, only "common" information are retained: This
    means that only keywords that exist in both dictionaries will be included
    in the resulting one.

    :type stats1: :class:`~obspy.core.trace.Stats`
    :param stats1: First Trace's stats
    :type stats2: :class:`~obspy.core.trace.Stats`
    :param stats2: Second Trace's stats
    :param start_lag: The lag of the first sample of the correlation given
        in seconds (usually negative).
    :type start_lag: float
    :type inv: :class:`~obspy.core.inventory.Inventory`, optional
    :param inv: Inventory containing the station coordinates. Only needed if
        station coordinates are not in Trace.Stats. Defaults to None.

    :rtype: :class:`~obspy.core.trace.Stats`
    :return: **stats**: combined Stats object
    """

    if not isinstance(stats1, Stats):
        raise TypeError("stats1 must be an obspy Stats object.")

    if not isinstance(stats2, Stats):
        raise TypeError("stats2 must be an obspy Stats object.")

    # We also have to remove these as they are obspy AttributeDicts as well
    stats1.pop('asdf', None)
    stats2.pop('asdf', None)

    tr1_keys = list(stats1.keys())
    tr2_keys = list(stats2.keys())

    stats = CorrStats()
    # actual correlation times
    stats['corr_start'] = max(stats1.starttime, stats2.starttime)
    stats['corr_end'] = min(stats1.endtime, stats2.endtime)

    # Adjust the information to create a new SEED like id
    keywords = ['network', 'station', 'location', 'channel']
    sac_keywords = ['sac']

    for key in keywords:
        if key in tr1_keys and key in tr2_keys:
            stats[key] = stats1[key] + '-' + stats2[key]

    for key in tr1_keys:
        if key not in keywords and key not in sac_keywords:
            if key in tr2_keys:
                if stats1[key] == stats2[key]:
                    # in the stats object there are read only objects
                    try:
                        stats[key] = stats1[key]
                    except (AttributeError, KeyError):
                        pass

    try:
        if ('stla' and 'stlo' and 'stel') in stats1:
            stats['stla'] = stats1.stla
            stats['stlo'] = stats1.stlo
            stats['stel'] = stats1.stel
            stats['evla'] = stats2.stla
            stats['evlo'] = stats2.stlo
            stats['evel'] = stats2.stel
        else:
            stats['stla'] = stats1.sac.stla
            stats['stlo'] = stats1.sac.stlo
            stats['stel'] = stats1.sac.stel
            stats['evla'] = stats2.sac.stla
            stats['evlo'] = stats2.sac.stlo
            stats['evel'] = stats2.sac.stel
            stats1.update(stats1['sac'])
            stats2.update(stats2['sac'])

        az, baz, dist = m3ut.trace_calc_az_baz_dist(stats1, stats2)

        stats['dist'] = dist / 1000
        stats['az'] = az
        stats['baz'] = baz
    except (AttributeError, KeyError):
        if inv:
            inv1 = inv.select(
                network=stats1.network, station=stats1.station)
            inv2 = inv.select(
                network=stats2.network, station=stats2.station)
            stats['stla'] = inv1[0][0].latitude
            stats['stlo'] = inv1[0][0].longitude
            stats['stel'] = inv1[0][0].elevation
            stats['evla'] = inv2[0][0].latitude
            stats['evlo'] = inv2[0][0].longitude
            stats['evel'] = inv2[0][0].elevation

            az, baz, dist = m3ut.inv_calc_az_baz_dist(inv1, inv2)

            stats['dist'] = dist / 1000
            stats['az'] = az
            stats['baz'] = baz
        else:
            warnings.warn("No station coordinates provided.")
    stats.pop('sac', None)
    stats.pop('response', None)
    stats['_format'] = 'hdf5'

    # note that those have to be adapted whenever several correlations are
    # stacked
    stats['start_lag'] = start_lag
    return stats


Compare_Str = "{network}.{station}.{channel}.{location}"
Compare_Str_No_Loc = "{network}.{station}.{channel}"


def compare_tr_id(tr0: Trace, tr1: Trace, regard_loc: bool = True) -> bool:
    """
    Check whether two traces are from the same channel, station, network, and,
    optionally, location. Useful for stacking

    :param tr0: first trace
    :type tr0: :class:`~obspy.core.trace.Trace`
    :param tr1: second trace
    :type tr1: :class:`~obspy.core.trace.Trace`
    :param regard_loc: Regard the location code or not
    :type regard_loc: bool
    :return: Bool whether the two are from the same (True) or not (False)
    :rtype: bool
    """
    if regard_loc:
        return Compare_Str.format(**tr0.stats)\
             == Compare_Str.format(**tr1.stats)
    else:
        return Compare_Str_No_Loc.format(**tr0.stats)\
             == Compare_Str_No_Loc.format(**tr1.stats)


def stack_st_by_group(st: Stream, regard_loc: bool, weight: str) -> CorrStream:
    """
    Stack all traces that belong to the same network, station, channel, and
    (optionally) location combination in the input stream.

    :param st: input Stream
    :type st: Stream
    :param regard_loc: Seperate data with different location code
    :type regard_loc: bool
    :return: :class:`~miic3.correlate.stream.CorrStream`
    :rtype: CorrStream
    """
    if regard_loc:
        key = "{network}.{station}.{channel}.{location}"
    else:
        key = "{network}.{station}.{channel}"
    stackdict = {}
    for tr in st:
        stackdict.setdefault(key.format(**tr.stats), CorrStream()).append(tr)
    stackst = CorrStream()
    for k in stackdict:
        stackst.append(stack_st(stackdict[k], weight))
    return stackst


def stack_st(st: CorrStream, weight: str, norm: bool = True) -> CorrTrace:
    """
    Returns an average of the data of all traces in the stream. Also adjusts
    the corr_start and corr_end parameters in the header.

    :param st: input Stream
    :type st: CorrStream
    :param weight: type of weigthing to use. Either `mean` or `by_length`
    :type weigth: str
    :param norm: Should the traces be normalised by their absolute maximum
        prior to stacking?
    :type norm: bool
    :return: Single trace with stacked data
    :rtype: CorrTrace
    """
    st.sort(keys=['corr_start'])
    stats = st[0].stats.copy()
    stats['corr_end'] = st[-1].stats['corr_end']
    st.sort(keys=['npts'])
    npts = st[-1].stats.npts
    stack = []
    dur = []  # duration of each trace
    for tr in st.select(npts=npts):
        stack.append(tr.data)
        dur.append(tr.stats.corr_end-tr.stats.corr_start)
    A = np.array(stack)
    if norm:
        norm = np.max(np.abs(A), axis=1)
        A /= np.tile(np.atleast_2d(norm).T, (1, A.shape[1]))
    if weight == 'mean' or weight == 'average':
        return CorrTrace(data=np.average(A, axis=0), _header=stats)
    elif weight == 'by_length':
        # Weight by the length of each trace
        data = np.sum((A.T*np.array(dur)).T, axis=0)/np.sum(dur)
        return CorrTrace(data=data, _header=stats)


def convert_statlist_to_bulk_stats(
        statlist: List[CorrStats], varying_loc: bool = False) -> CorrStats:
    """
    Converts a list of :class:`~miic3.correlate.stream.CorrTrace` stats objects
    to a single stats object that can be used for the creation of a
    :class:`~miic3.correlate.stream.CorrBulk` object

    :param statlist: list of Stats
    :type statlist: List[Stats]
    :param varying_loc: Set true if the location codes vary
    :type varying_loc: False
    :raises ValueError: raised if data does not fit together
    :return: single Stats object
    :rtype: Stats
    """
    stats = statlist[0].copy()
    # can change from trace to trace
    mutables = ['corr_start', 'corr_end']

    # Should / have to be identical for each trace
    # Not 100% sure if start and end_lag should be on this list
    immutables = [
        'npts', 'sampling_rate', 'network', 'station', 'channel', 'start_lag',
        'end_lag', 'stla', 'stlo', 'stel', 'evla', 'evlo', 'evel',
        'dist', 'az', 'baz']
    if varying_loc:
        mutables += ['location']
    else:
        immutables += ['location']
    for key in mutables:
        stats[key] = []
    for trstat in statlist:
        for key in mutables:
            stats[key] += [trstat[key]]
        for key in immutables:
            if stats[key] != trstat[key]:
                raise ValueError('The stream contains data with different \
properties. The differing property is %s' % key)
    stats['ntrcs'] = len(statlist)
    return stats
