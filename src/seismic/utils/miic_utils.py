'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 29th March 2021 12:54:05 pm
Last Modified: Tuesday, 1st March 2022 10:04:56 am
'''
from typing import List, Tuple
import logging
import re
import warnings

import numpy as np
from obspy import Inventory, Stream, Trace, UTCDateTime
from obspy.core import Stats, AttribDict

# zero lag time
# I don't think this is in use anymore
lag0 = UTCDateTime(0)

log_lvl = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR}


def trace_calc_az_baz_dist(stats1: Stats, stats2: Stats) -> Tuple[
        float, float, float]:
    """
    Return azimuth, back azimhut and distance between tr1 and tr2
    This funtions calculates the azimut, back azimut and distance between tr1
    and tr2 if both have geo information in their stats dictonary.
    Required fields are:
    ``tr.stats.sac.stla`` and ``tr.stats.sac.stlo``

    :type stats1: :class:`~obspy.core.Stats`
    :param stats1: First trace to account
    :type stats2: :class:`~obspy.core.Stats`
    :param stats2: Second trace to account

    :rtype: float
    :return: **az**: Azimuth angle between tr1 and tr2
    :rtype: float
    :return: **baz**: Back-azimuth angle between tr1 and tr2
    :rtype: float
    :return: **dist**: Distance between tr1 and tr2 in m
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

    dist, az, baz = gps2dist_azimuth(
        stats1['stla'], stats1['stlo'], stats2['stla'], stats2['stlo'])

    return az, baz, dist


def filter_stat_dist(inv1: Inventory, inv2: Inventory, thres: float) -> bool:
    """
    Very simple function to check whether to stations are closer than thres
    to each other.

    :param inv1: Inventory of station 1
    :type inv1: Inventory
    :param inv2: Inventory of station 2
    :type inv2: Inventory
    :param thres: Threshold distance in m
    :type thres: float
    :return: True if closer (or equal) than thres, False if not.
    :rtype: bool
    """
    return inv_calc_az_baz_dist(inv1, inv2)[-1] <= thres


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
    :return: **dist**: Distance between stat1 and stat2 in m
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

    dist, az, baz = gps2dist_azimuth(
        inv1[0][0].latitude, inv1[0][0].longitude, inv2[0][0].latitude,
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
    # Decimation factor
    factor = float(sr)/float(srn)
    if filter and factor <= 16:
        freq = sr * 0.5 / factor
        data.filter('lowpass_cheby_2', freq=freq, maxorder=12)
    elif filter:
        # Use a different filter
        freq = sr * 0.45 / factor
        data.filter('lowpass_cheby_2', freq=freq, maxorder=12)

    if sr/srn == sr//srn:
        return data.decimate(int(sr//srn), no_filter=True)
    else:
        return data.resample(srn)


def trim_stream_delta(
        st: Stream, start: float, end: float, *args, **kwargs) -> Stream:
    """
    Cut all traces to starttime+start and endtime-end. *args* and *kwargs* will
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
    Cut all traces to starttime+start and endtime-end. *args* and *kwargs* will
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


def load_header_from_np_array(array_dict: dict) -> dict:
    """
    Takes the *dictionary-like* return-value of `np.load` and converts the
    corresponding keywords into an obspy header.

    :param array_dict: Return value of `np.load`
    :type array_dict: dict
    :return: The obspy header as dictionary
    :rtype: dict
    """
    d = {}
    for k in array_dict:
        if k in no_stats or re.match('reftr', k):
            continue
        elif k in t_keys:
            d[k] = convert_timestamp_to_utcdt(array_dict[k])
        else:
            try:
                d[k] = array_dict[k][0]
            except IndexError:
                warnings.warn(
                    f'Key {k} could not be loaded into the header.')
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


def get_valid_traces(st: Stream):
    """Return only valid traces of a stream.

    Remove traces that are 100% masked from a stream. This happens when
    a masked trace is trimmed within a gap. The function works in place.

    :type st: obspy.Stream
    :param st: stream to work on

    """
    for tr in st:
        if isinstance(tr.data, np.ma.MaskedArray):
            if tr.data.mask.all():
                st.remove(tr)
    return


def discard_short_traces(st: Stream, length: float):
    """
    Discard all traces from stream that are shorter than length.

    :param st: inputer obspy Stream
    :type st: Stream
    :param length: Maxixmum Length that should be discarded (in seconds).
    :type length: float

    .. note:: Action is performed in place.
    """
    for tr in st:
        if tr.stats.npts/tr.stats.sampling_rate <= length:
            st.remove(tr)
            logging.debug(f'Discarding short Trace {tr}.')
    return


def nan_moving_av(
        data: np.ndarray, win_half_len: int, axis: int = -1) -> np.ndarray:
    """
    Returns a filtered version of data, disregarding the nans.
    Moving mean window length is win_half_len*2+1.

    :param data: Array to be filtered
    :type data: np.ndarray
    :param win_half_len: Half length of the boxcar filter (len = halflen*2+1)
    :type win_half_len: int
    :param axis: Axis to filter along, defaults to -1
    :type axis: int, optional
    :return: The filtered array
    :rtype: np.ndarray
    """
    # Swap axes, so we can work on queried axis
    dataswap = data.swapaxes(0, axis)
    data_smooth = np.empty_like(dataswap)
    for ii in range(dataswap.shape[0]):
        start = ii - win_half_len
        if start < 0:
            start = 0
        # weighted average
        data_smooth[ii] = np.nanmean(
            dataswap[start:ii+win_half_len+1], axis=0)
    return data_smooth.swapaxes(0, axis)


def stream_require_dtype(st: Stream, dtype: type) -> Stream:
    """
    Often it might make sense to change the data type of seismic data before
    saving or broadcasting (e.g., to save memory). This function allows
    to do so

    :param st: input Stream
    :type st: Stream
    :param dtype: desired datatype, e.g. np.float32
    :type dtype: type
    :return: Stream with data in new dtype
    :rtype: Stream

    .. note:: This operation is performed in place.
    """
    for tr in st:
        tr.data = np.require(tr.data, dtype)
