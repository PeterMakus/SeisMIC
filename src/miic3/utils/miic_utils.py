'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 29th March 2021 12:54:05 pm
Last Modified: Tuesday, 6th July 2021 05:54:19 pm
'''
from typing import List, Tuple
from warnings import warn

import numpy as np
from obspy import Inventory, Stream, Trace, UTCDateTime
from obspy.core import Stats, AttribDict


# zero lag time
lag0 = UTCDateTime(0)


def trace_calc_az_baz_dist(stats1: Stats, stats2: Stats) -> Tuple[
        float, float, float]:
    """ Return azimuth, back azimhut and distance between tr1 and tr2
    This funtions calculates the azimut, back azimut and distance between tr1
    and tr2 if both have geo information in their stats dictonary.
    Required fields are:
        tr.stats.sac.stla
        tr.stats.sac.stlo

    :type stats1: :class:`~obspy.core.Stats`
    :param stats1: First trace to account
    :type stats2: :class:`~obspy.core.Stats`
    :param stats2: Second trace to account

    :rtype: float
    :return: **az**: Azimuth angle between tr1 and tr2
    :rtype: float
    :return: **baz**: Back-azimuth angle between tr1 and tr2
    :rtype: float
    :return: **dist**: Distance between tr1 and tr2
    """

    if not isinstance(stats1, (Stats, AttribDict)):
        raise TypeError("stats1 must be an obspy Stats object.")

    if not isinstance(stats2, (Stats, AttribDict)):
        raise TypeError("stats2 must be an obspy Stats object.")

    try:
        from obspy.geodetics import gps2dist_azimuth
    except ImportError:
        print("Missed obspy funciton gps2dist_azimuth")
        print("Update obspy.")
        return

    dist, az, baz = gps2dist_azimuth(stats1.sac.stla,
                                     stats1.sac.stlo,
                                     stats2.sac.stla,
                                     stats2.sac.stlo)

    return az, baz, dist


def inv_calc_az_baz_dist(inv1: Inventory, inv2: Inventory) -> Tuple[
        float, float, float]:
    """ Return azimuth, back azimuth and distance between stat1 and stat2


    :type tr1: :class:`~obspy.core.inventory.Inventory`
    :param tr1: First trace to account
    :type tr2: :class:`~obspy.core.inventory.Inventory`
    :param tr2: Second trace to account

    :rtype: float
    :return: **az**: Azimuth angle between stat1 and stat2
    :rtype: float
    :return: **baz**: Back-azimuth angle between stat2 and stat2
    :rtype: float
    :return: **dist**: Distance between stat1 and stat2
    """

    if not isinstance(inv1, Inventory):
        raise TypeError("inv1 must be an obspy Inventory.")

    if not isinstance(inv2, Inventory):
        raise TypeError("inv2 must be an obspy Inventory.")

    try:
        from obspy.geodetics import gps2dist_azimuth
    except ImportError:
        print("Missing obspy funciton gps2dist_azimuth")
        print("Update obspy.")
        return

    dist, az, baz = gps2dist_azimuth(inv1[0][0].latitude,
                                     inv1[0][0].longitude,
                                     inv2[0][0].latitude,
                                     inv2[0][0].longitude)

    return az, baz, dist


def resample_or_decimate(
    data: Trace or Stream, sampling_rate_new: int,
        filter=True) -> Stream or Trace:
    """Decimates the data if the desired new sampling rate allows to do so.
    Else the signal will be interpolated (a lot slower).

    :param data: Stream to be resampled.
    :type data: Stream
    :param sampling_rate_new: The desired new sampling rate
    :type sampling_rate_new: int
    :return: The resampled stream
    :rtype: Stream
    """
    if isinstance(data, Stream):
        sr = data[0].stats.sampling_rate
    elif isinstance(data, Trace):
        sr = data.stats.sampling_rate
    else:
        raise TypeError('Data has to be an obspy Stream or Trace.')

    srn = sampling_rate_new
    if srn > sr:
        raise ValueError('New sampling rate greater than old. This function \
            is only intended for downsampling.')
    elif srn == sr:
        return data

    # Chosen this filter design as it's exactly the same as
    # obspy.Stream.decimate uses
    if filter:
        freq = sr * 0.5 / float(sr/srn)
        data.filter('lowpass_cheby_2', freq=freq, maxorder=12)

    if sr/srn == sr//srn:
        return data.decimate(int(sr//srn), no_filter=True)
    else:
        return data.resample(srn)


def stream_filter(st: Stream, ftype: str, filter_option: dict) -> Stream:
    """ Filter each trace of a Stream according to the given parameters

    This faction apply the specified filter function to all the traces in the
    present in the input :py:class:`~obspy.core.stream.Stream`.

    :type ftype: str
    :param ftype: String that specifies which filter is applied (e.g.
            ``"bandpass"``). See the `Supported Filter`_ section below for
            further details.
    :type filter_option: dict
    :param filter_option: Necessary arguments for the respective filter
        that will be passed on. (e.g. ``freqmin=1.0``, ``freqmax=20.0`` for
        ``"bandpass"``)
    :type parallel: bool (Default: True)
    :pram parallel: If the filtering will be run in parallel or not
    :type processes: int
    :pram processes: Number of processes to start (if None it will be equal
        to the number of cores available in the hosting machine)

    .. note::

        This operation is performed in place on the actual data arrays. The
        raw data is not accessible anymore afterwards. To keep your
        original data, use :func:`~miic.core.alpha_mod.stream_copy` to create
        a copy of your stream object.
        This function can also work in parallel an all or a specified number of
        cores available in the hosting machine.

    .. rubric:: _`Supported Filter`

    ``'bandpass'``
        Butterworth-Bandpass (uses :func:`obspy.signal.filter.bandpass`).

    ``'bandstop'``
        Butterworth-Bandstop (uses :func:`obspy.signal.filter.bandstop`).

    ``'lowpass'``
        Butterworth-Lowpass (uses :func:`obspy.signal.filter.lowpass`).

    ``'highpass'``
        Butterworth-Highpass (uses :func:`obspy.signal.filter.highpass`).

    ``'lowpassCheby2'``
        Cheby2-Lowpass (uses :func:`obspy.signal.filter.lowpassCheby2`).

    ``'lowpassFIR'`` (experimental)
        FIR-Lowpass (uses :func:`obspy.signal.filter.lowpassFIR`).

    ``'remezFIR'`` (experimental)
        Minimax optimal bandpass using Remez algorithm (uses
        :func:`obspy.signal.filter.remezFIR`).

    """
    if not isinstance(st, Stream):
        raise TypeError("'st' must be a 'obspy.core.stream.Stream' object")

    fparam = dict(
        [(kw_filed, filter_option[kw_filed]) for kw_filed in filter_option])

    # take care of masked traces and keep their order in the stream
    fst = Stream()
    for tr in st:
        sptr = tr.split()
        sptr.filter(ftype, **fparam)
        sptr.merge()
        fst += sptr
    st = fst

    # Change the name to help blockcanvas readability
    st_filtered = st
    return st_filtered


def detrend_st(st, *args, **kwargs):
    out = Stream()
    for tr in st:
        sst = tr.split()
        sst.detrend(*args, **kwargs)
        out.extend(sst.merge())
        st.remove(tr)
    return out


def cos_taper_st(
        st: Stream, taper_len: float, taper_at_masked: bool) -> Stream:
    """
    Applies a cosine taper to the input Stream.

    :param tr: Input Stream
    :type tr: :class:`~obspy.core.stream.Stream`
    :param taper_len: Length of the taper per side
    :type taper_len: float
    :param taper_at_masked: applies a split to each trace and merges again
        afterwards
    :type taper_at_masked: bool
    :return: Tapered Stream
    :rtype: :class:`~obspy.core.stream.Stream`

    .. note::
        This action is performed in place. If you want to keep the
        original data use :func:`~obspy.core.stream.Stream.copy`.
    """
    if isinstance(st, Trace):
        st = [st]
    # out = Stream()
    for tr in st:
        try:
            cos_taper(tr, taper_len, taper_at_masked)
        except ValueError as e:
            warn('%s, corresponding trace will be removed.' % e)
            st.remove(tr)
    return st


def cos_taper(tr: Trace, taper_len: float, taper_at_masked: bool) -> Trace:
    """
    Applies a cosine taper to the input trace.

    :param tr: Input Trace
    :type tr: Trace
    :param taper_len: Length of the taper per side in seconds
    :type taper_len: float
    :param taper_at_masked: applies a split to each trace and merges again
        afterwards
    :type taper_at_masked: bool
    :return: Tapered Trace
    :rtype: Trace

    .. note::

        This action is performed in place. If you want to keep the
        original data use :func:`~obspy.core.trace.Trace.copy`.
    """
    if taper_len <= 0:
        raise ValueError('Taper length must be larger than 0 s')
    if taper_at_masked:
        st = tr.split()
        st = cos_taper_st(st, taper_len, False)
        st = st.merge()
        if st.count():
            return st[0]
        else:
            raise ValueError('Taper length must be larger than 0 s')
    taper = np.ones_like(tr.data)
    tl_n = round(taper_len*tr.stats.sampling_rate)
    if tl_n * 2 > tr.stats.npts:
        raise ValueError(
            'Taper Length * 2 has to be smaller or equal to trace\'s length.')
    tap = np.sin(np.linspace(0, np.pi, tl_n*2))
    taper[:tl_n] = tap[:tl_n]
    taper[-tl_n:] = tap[-tl_n:]
    tr.data = np.multiply(tr.data, taper)
    return tr


def trim_stream_delta(
        st: Stream, start: float, end: float, *args, **kwargs) -> Stream:
    """
    Cut all traces to starttime+start and endtime-end. *args and **kwargs will
    be passed to :func:`~obspy.Stream.trim`

    :param st: Input Stream
    :type st: Stream
    :param start: Delta to add to old starttime (in seconds)
    :type start: float
    :param end: Delta to remove from old endtime (in seconds)
    :type end: float
    :return: The trimmed stream
    :rtype: Stream

    .. note::

        This operation is performed in place on the actual data arrays.
        The raw data will no longer be accessible afterwards. To keep your
        original data, use copy() to create a copy of your stream object.

    """
    for tr in st:
        tr = trim_trace_delta(tr, start, end, *args, **kwargs)
    return st


def trim_trace_delta(
        tr: Trace, start: float, end: float, *args, **kwargs) -> Trace:
    """
    Cut all traces to starttime+start and endtime-end. *args and **kwargs will
    be passed to :func:`~obspy.Trace.trim`.

    :param st: Input Trace
    :type st: Trace
    :param start: Delta to add to old starttime (in seconds)
    :type start: float
    :param end: Delta to remove from old endtime (in seconds)
    :type end: float
    :return: The trimmed trace
    :rtype: Trace

    .. note::

        This operation is performed in place on the actual data arrays.
        The raw data will no longer be accessible afterwards. To keep your
        original data, use copy() to create a copy of your stream object.

    """
    return tr.trim(
        starttime=tr.stats.starttime+start, endtime=tr.stats.endtime-end,
        *args, **kwargs)


# Time keys
t_keys = ['starttime', 'endtime', 'corr_start', 'corr_end']
# No stats, keys that are not in stats but attributes of the respective objects
no_stats = [
    'corr', 'value', 'sim_mat', 'second_axis', 'method_array', 'vt_array',
    'data']


def save_header_to_np_array(stats: Stats) -> dict:
    """
    Converts an obspy header to a format that allows it to be saved in an
    npz file (i.e., several gzipped npy files)

    :param stats: input Header
    :type stats: Stats
    :return: Dictionary, whose keys are the names of the arrays and the values
        the arrays themselves. Can be fed into ``np.savez`` as `**kwargs`
    :rtype: dict
    """
    array_dict = {}
    for k in stats:
        if k in t_keys:
            array_dict[k] = convert_utc_to_timestamp(stats[k])
        else:
            array_dict[k] = np.array([stats[k]])
    return array_dict


def load_header_from_np_array(array_dict: dict) -> Stats:
    """
    Takes the *dictionary-like* return-value of `np.load` and converts the
    corresponding keywords into an obspy header.

    :param array_dict: Return value of `np.load`
    :type array_dict: dict
    :return: The obspy header object
    :rtype: Stats
    """
    d = {}
    for k in array_dict:
        if k in no_stats:
            continue
        elif k in t_keys:
            d[k] = convert_timestamp_to_utcdt(array_dict[k])
        else:
            d[k] = array_dict[k][0]
    return d


def convert_utc_to_timestamp(
        utcdt: UTCDateTime or List[UTCDateTime]) -> np.ndarray:
    """
    Converts :class:`obspy.core.utcdatetime.UTCDateTime` objects to floats.

    :param utcdt: The input times, either a list of utcdatetimes or one
        utcdatetime
    :type utcdt: UTCDateTimeorList[UTCDateTime]
    :return: A numpy array of timestamps
    :rtype: np.ndarray
    """
    if isinstance(utcdt, UTCDateTime):
        utcdt = [utcdt]
    timestamp = np.array([t.timestamp for t in utcdt])
    return timestamp


def convert_timestamp_to_utcdt(timestamp: np.ndarray) -> List[UTCDateTime]:
    """
    Converts a numpy array holding timestamps (i.e., floats) to a list of
    UTCDateTime objects

    :param timestamp: numpy array holding timestamps
    :type timestamp: np.ndarray
    :return: a list of UTCDateTime objects
    :rtype: List[UTCDateTime]
    """
    timestamp = list(timestamp)
    for ii, t in enumerate(timestamp):
        timestamp[ii] = UTCDateTime(t)
    if len(timestamp) == 1:
        timestamp = timestamp[0]
    return timestamp
