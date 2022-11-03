'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Christoph Sens-Schönefelder
    Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 14th June 2021 08:50:57 am
Last Modified: Thursday, 3rd November 2022 10:46:08 am
'''

from typing import List, Tuple
import numpy as np
from copy import deepcopy
from scipy.signal import butter, lfilter, hilbert, resample
from scipy.interpolate import UnivariateSpline
from obspy.core import UTCDateTime

# Obspy imports
from obspy.signal.invsim import cosine_taper

from seismic.monitor.stretch_mod import multi_ref_vchange_and_align, \
    time_shift_estimate, compare_with_modified_reference, \
    create_shifted_ref_mat
from seismic.correlate.stats import CorrStats
from seismic.monitor.dv import DV


def corr_mat_clip(A: np.ndarray, thres: float, axis: int) -> np.ndarray:
    """
    Clip the Input Matrix upper and lower bounds to a multiple of its
    standard deviation. `thres` determines the factor and `axis` the axis
    the std should be computed over

    :param A: Input Array to be clipped
    :type A: np.ndarray
    :param thres: factor of the standard deviation to clip by
    :type thres: float
    :param axis: Axis to compute the std over and, subsequently clip over.
        Can be None, if you wish to compute floating point rather than a
        vector. Then, the array will be clipped evenly.
    :type axis: int
    :return: The clipped array
    :rtype: np.ndarray
    """
    clipv = thres * np.std(A, axis=axis)
    if axis == 1:
        A = np.clip(A.T, -clipv, clipv).T
    else:
        A = np.clip(A, -clipv, clipv)
    return A


def _smooth(
    x: np.ndarray, window_len: int = 10,
        window: str = 'hanning') -> np.ndarray:
    """ Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    :type x: :class:`~numpy.ndarray`
    :param x: the input signal
    :type window_len: int
    :param window_len: the dimension of the smoothing window
    :type window: string
    :param window: the type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman' flat window will produce a moving average
        smoothing.

    :rtype: :class:`~numpy.ndarray`
    :return: The smoothed signal

    >>>import numpy as np
    >>>t = np.linspace(-2,2,0.1)
    >>>x = np.sin(t)+np.random.randn(len(t))*0.1
    >>>y = smooth(x)

    .. rubric:: See also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve, scipy.signal.lfilter

    TODO: the window parameter could be the window itself if it is an array
        instead of a string
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming',\
            'bartlett', 'blackman'")

    s = np.r_[2 * x[0] - x[window_len:1:-1], x,
              2 * x[-1] - x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')

    return y[window_len - 1:-window_len + 1]


def corr_mat_smooth(
    data: np.ndarray, wsize: int, wtype: str = 'flat',
        axis: int = 1) -> np.ndarray:
    """ Smoothing of a correlation matrix.

    Smoothes the correlation matrix with a given window function of the given
    width along the given axis. This method is based on the convolution of a
    scaled window with the signal. Each row/col (i.e. depending on the selected
    ``axis``) is "prepared" by introducing reflected copies of it (with the
    window size) in both ends so that transient parts are minimized in the
    beginning and end part of the resulting array.

    :type corr_mat: dictionary of the type correlation matrix
    :param corr_mat: correlation matrix to be smoothed
    :type wsize: int
    :param wsize: Window size
    :type wtype: string
    :param wtype: Window type. It can be one of:
            ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'] defaults to
            'flat'
    :type axis: int
    :param axis: Axis along with apply the filter. O: smooth along correlation
              lag time axis 1: smooth along time axis

    :rtype: :class:`~numpy.ndarray`
    :return: **X**: Filtered matrix
    """
    # Remove nans
    np.nan_to_num(data, copy=False)

    # Degenerated corr_mat: single vector
    try:
        row, col = data.shape
    except ValueError:
        # Single vector not a matrix
        data = _smooth(data, window_len=wsize, window=wtype)
        return data

    # Proper 2d matrix. Smooth on the chosen axis
    if axis == 0:
        for i in np.arange(row):
            csig = data[i]
            scsig = _smooth(csig, window_len=wsize, window=wtype)
            data[i] = scsig
    elif axis == 1:
        for i in np.arange(col):
            csig = data[:, i]
            scsig = _smooth(csig, window_len=wsize, window=wtype)
            data[:, i] = scsig

    return data


def unicode_to_string(input):
    """Convert all unicode strings to utf-8 strings
    """
    if isinstance(input, dict):
        return {
            unicode_to_string(key): unicode_to_string(
                value) for key, value in input.items()}
    elif isinstance(input, list):
        return [unicode_to_string(element) for element in input]
    elif isinstance(input, str):
        return input.encode('utf-8')
    else:
        return input


def corr_mat_filter(
    data: np.ndarray, stats: CorrStats, freqs: Tuple[float, float],
        order=3) -> np.ndarray:
    """ Filter a correlation matrix.

    Filters the correlation matrix corr_mat in the frequency band specified in
    freqs using a zero phase filter of twice the order given in order.

    :type data: np.ndarray
    :param data: correlation matrix
    :type stats: :class:`~seismic.correlate.stats.CorrStats`
    :param stats: The stats object corresponding to the CorrBulk object
    :type freqs: array-like of length 2
    :param freqs: lower and upper limits of the pass band in Hertz
    :type order: int
    :param order: half the order of the Butterworth filter

    :rtype: np.ndarray
    :return: filtered correlation matrix
    """

    if len(freqs) != 2:
        raise ValueError("freqs needs to be a two element array with the \
            lower and upper limits of the filter band in Hz.")

    # # end check

    fe = float(stats['sampling_rate']) / 2

    (b, a) = butter(
        order, np.array(freqs, dtype='float') / fe, btype='band')

    data = lfilter(b, a, data, axis=1)
    data = lfilter(b, a, data[:, ::-1], axis=1)[:, ::-1]

    return data


def corr_mat_trim(
    data: np.ndarray, stats: CorrStats, starttime: float,
        endtime: float) -> Tuple[np.ndarray, CorrStats]:
    """ Trim the correlation matrix to a given period.

    Trim the correlation matrix `corr_mat` to the period from `starttime` to
    `endtime` given in seconds from the zero position, so both can be
    positive and negative. If `starttime` and `endtime` are datetime.datetime
    objects they are taken as absolute start and endtime.

    :type data: np.ndarray
    :param corr_mat: correlation matrix to be trimmed
    :type starttime: float
    :param starttime: start time in seconds with respect to the zero position.
        Hence, this parameter describes a lag!
    :type endtime: float
    :param order: end time in seconds with respect to the zero position

    :rtype tuple: Tuple[np.ndarray, CorrStats]
    :return: trimmed correlation matrix and new stats object
    """

    # fetch indices of start and end
    start = int(
        np.floor((starttime-stats['start_lag'])*stats['sampling_rate']))
    end = int(np.floor((endtime-stats['start_lag'])*stats['sampling_rate']))

    # check range
    if start < 0:
        print('Error: starttime before beginning of trace. Data not changed')
        return data, stats
    if end >= stats['npts']:
        print('Error: endtime after end of trace. Data not changed')
        return data, stats

    # select requested part from matrix
    # +1 is to include the last sample
    if len(data.shape) == 1:
        data = data[start:end+1]
        stats['npts'] = len(data)
    else:
        data = data[:, start:end + 1]
        stats['npts'] = data.shape[1]

    # set starttime, endtime and npts of the new stats
    stats['start_lag'] = starttime

    return data, stats


def corr_mat_resample(
    data: np.ndarray, stats: CorrStats, start_times: List[UTCDateTime],
        end_times=[]) -> Tuple[np.ndarray, CorrStats]:
    """ Function to create correlation matrices with constant sampling

    When created with recombine_corr_data the correlation matrix contains all
    available correlation streams but homogeneous sampling is not guaranteed
    as correlation functions may be missing due to data gaps. This function
    restructures the correlation matrix by inserting or averaging correlation
    functions to provide temporally homogeneous sampling. Inserted correlation
    functions consist of 'nan' if gaps are present and averaging is done if
    more than one correlation function falls in a bin between start_times[i]
    and end_times[i]. If end_time is an empty list (default) end_times[i] is
    set to start_times[i] + (start_times[1] - start_times[0])

    :type data: np.ndarray
    :param data: 2D matrix holding the correlation data
    :type Stats: :class:`~obspy.core.Stats`
    :param Stats: The stats object belonging to the
        :class:`~seismic.correlate.stream.CorrBulk` object.
    :type start_times: list of :class:`~obspy.UTCDateTime` objects
    :param start_times: list of starting times for the bins of the new
        sampling
    :type end_times: list of :class:`~obspy.UTCDateTime` objects
    :param end_times: list of starting times for the bins of the new sampling

    :rtype: tuple
    :return: the new data array and altered stats object.
    """

    if len(end_times) > 0 and (len(end_times) != len(start_times)):
        raise ValueError("end_times should be empty or of the same length as \
            start_times.")

    # old sampling times
    otime = np.array([ii.timestamp for ii in stats['corr_start']])
    if isinstance(start_times[0], UTCDateTime):
        start_times = [ii.timestamp for ii in start_times]

    # new sampling times
    stime = start_times
    if len(end_times):
        if isinstance(end_times[0], UTCDateTime):
            end_times = [ii.timestamp for ii in end_times]
        etime = end_times
    else:
        if len(start_times) == 1:
            # there is only one start_time given and no end_time => average all
            etime = np.array([stats['corr_end'][-1].timestamp])
        else:
            stime = np.array(stime)  # Cannot be a list for this operation
            etime = stime + (stime[1] - stime[0])

    # create masked array to avoid nans
    mm = np.ma.masked_array(
        data, np.isnan(data))

    # new corr_data matrix
    nmat = np.empty([len(stime), data.shape[1]])
    nmat.fill(np.nan)

    for ii in range(len(stime)):
        # index of measurements between start_time[ii] and end_time[ii]
        ind = np.nonzero((otime >= stime[ii]) * (otime < etime[ii]))  # ind is
        # a list(tuple) for dimensions
        if len(ind[0]) == 1:
            # one measurement found
            nmat[ii, :] = data[ind[0], :]
        elif len(ind[0]) > 1:
            # more than one measurement in range
            nmat[ii, :] = np.mean(mm[ind[0], :], 0).filled(np.nan)

    # assign new data
    data = nmat

    stats['corr_start'] = [UTCDateTime(st) for st in stime]
    stats['corr_end'] = [UTCDateTime(et) for et in etime]

    return data, stats


# Probably not necessary
def corr_mat_correct_decay(data: np.ndarray, stats: CorrStats) -> np.ndarray:
    """
    Correct for the amplitude decay in a correlation matrix.

    Due to attenuation and geometrical spreading the amplitude of the
    correlations decays with increasing lapse time. This decay is corrected
    by dividing the correlation functions by an exponential function that
    models the decay.

    :param data: Matrix holding correlation data
    :type data: np.ndarray
    :param stats: Stats object afiliated to a CorrBulk object
    :type stats: CorrStats
    :return: correlations with decay corrected amplitudes
    :rtype: np.ndarray
    """
    # copy input
    # cmat = deepcopy(corr_mat)

    # calculate the envelope of the correlation matrix
    env = corr_mat_envelope(data)

    # average causal and acausal part of the correlation matrix
    env, stats_mirr = corr_mat_mirror(env, stats)

    # average to a single correlation function
    env = np.nanmean(env, 0)  # Shouldn't that be a mean or divided by the dim?
    # env = np.nansum(env, 0)

    # fit the decay
    t = np.arange(stats_mirr['npts'])
    le = np.log(env)
    K, log_A = np.polyfit(t, le, 1)
    A = np.exp(log_A)

    # central sample
    cs = -stats['start_lag']*stats['sampling_rate']
    # time vector
    t = np.absolute(np.arange(stats['npts']) - cs)
    # theoretical envelope
    tenv = A * np.exp(K * t)

    # correct with theoretical envelope
    data /= np.tile(tenv, [data.shape[0], 1])

    return data


def corr_mat_envelope(data: np.ndarray) -> np.ndarray:
    """ Calculate the envelope of a correlation matrix.

    The corrlation data of the correlation matrix are replaced by their
    Hilbert envelopes.

    :type data: np.ndarray
    :param data: correlation matrix.

    :rtype: np.ndarray
    :return: **hilb**: The envelope of the correlation data in the same shape
        as the input array.
    """

    # calculate the Hilbert transform of the correlation data
    hilb = hilbert(data, axis=1)

    # replace corr_data with their envelopes
    hilb = np.absolute(hilb)

    return hilb


def corr_mat_normalize(
    data: np.ndarray, stats: CorrStats, starttime: float = None,
        endtime: float = None, normtype: str = 'energy') -> np.ndarray:
    """ Correct amplitude variations with time in a correlation matrix.

    Measure the maximum of the absolute value of the correlation matrix in
    a specified lapse time window and normalize the correlation traces by
    this values. A coherent phase in the respective lapse time window will
    have constant ampitude afterwards..

    :type data: np.ndarray
    :param data: correlation matrix from CorrBulk
    :type stats: :class:`~seismic.correlate.stats.CorrStats`
    :param stats: The stats object from the
        :class:`~seismic.correlate.stream.CorrBulk` object.
    :type starttime: float
    :param starttime: beginning of time window in seconds with respect to the
        zero position
    :type endtime: float
    :param endtime: end time window in seconds with respect to the zero
        position
    :type normtype: string
    :param normtype: one of the following 'energy', 'max', 'absmax', 'abssum'
        to decide about the way to calculate the normalization.

    :rtype: np.ndarray
    :return: Correlation matrix with normalized ampitudes.
    """

    # calculate indices of the time window
    # start
    # needs to be like this because this condition also has to work when it's 0
    if starttime is not None:
        # first sample
        start = starttime - stats['start_lag']
        start = int(np.floor(start*stats['sampling_rate']))
    else:
        start = None

    if endtime is not None:
        end = endtime - stats['start_lag']
        end = int(np.floor(end*stats['sampling_rate']))
    else:
        end = None

    # check range
    if start and start < 0:
        print('Error: starttime before beginning of trace. Data not changed.')
        return data
    if end and end >= stats['npts']:
        print('Error: endtime after end of trace. Data not changed.')
        return data

    # calculate normalization factors
    if normtype == 'energy':
        norm = np.sqrt(np.mean(data[:, start:end] ** 2, 1))
    elif normtype == 'abssum':
        norm = np.mean(np.abs(data[:, start:end]), 1)
    elif normtype == 'max':
        norm = np.max(data[:, start:end], axis=1)
    elif normtype == 'absmax':
        norm = np.max(np.abs(data[:, start:end]), axis=1)
    else:
        raise ValueError('Normtype is unknown.')

    # normalize the matrix
    norm[norm == 0] = 1
    data /= np.tile(np.atleast_2d(norm).T, (1, data.shape[1]))
    return data


def corr_mat_mirror(data: np.ndarray, stats: CorrStats) -> Tuple[
        np.ndarray, CorrStats]:
    """
    Average the causal and acausal parts of a correlation matrix.

    :param data: A matrix holding correlation data
    :type data: np.ndarray
    :param stats: the stats object corresponding to a CorrBulk object.
    :type stats: CorrStats
    :return: as the input but with
        avaraged causal and acausal parts of the correlation data
    :rtype: Tuple[np.ndarray, CorrStats]
    """

    zero_sample = -(stats['start_lag']*stats['sampling_rate'])

    if zero_sample <= 0:
        print('No data present for mirroring: starttime > zerotime.')
        return data, stats
    if stats['end_lag'] <= 0:
        print('No data present for mirroring: endtime < zerotime.')
        return data, stats
    if np.fmod(zero_sample, 1) != 0:
        print('need to shift for mirroring')
        return 0

    # estimate size of mirrored array
    acausal_samples = int(zero_sample + 1)
    causal_samples = int(stats['npts'] - acausal_samples + 1)
    # +1 because sample a zerotime counts twice
    size = np.max([acausal_samples, causal_samples])
    both = np.min([acausal_samples, causal_samples])

    # allocate array
    mir_data = np.zeros([data.shape[0], size])

    # fill the array
    mir_data[:, 0:causal_samples] = data[:, acausal_samples - 1:]
    mir_data[:, 0:acausal_samples] += data[:, acausal_samples - 1::-1]

    # divide by two where both are present
    mir_data[:, 0:both] /= 2.

    # adopt the stats
    mir_stats = stats.copy()
    mir_stats['start_lag'] = 0
    mir_stats['npts'] = size
    # -1 because of zero sample
    # mir_stats['end_lag'] = (size-1)/mir_stats['sampling_rate']

    return mir_data, mir_stats


def corr_mat_taper(
        data: np.ndarray, stats: CorrStats, width: float) -> np.ndarray:
    """ Taper a correlation matrix.

    Apply a taper to all traces in the correlation matrix.

    :type data: np.ndarray
    :param data: correlation matrix`
    :type width: float
    :param width: width to be tapered in seconds (per side)

    :rtype: np.ndarray
    :return: The tapered matrix

    ..note:: **In-place operation**
    """
    if width == 0:
        return data
    elif width > stats['npts']/stats['sampling_rate']:
        raise ValueError('Taper longer than signal.')
    width *= 2
    # calculate taper
    tap = cosine_taper(
        stats['npts'], width*stats['sampling_rate']/stats['npts'])

    # apply taper
    data *= np.tile(np.atleast_2d(tap), (data.shape[0], 1))

    return data


def corr_mat_taper_center(
    data: np.ndarray, stats: CorrStats, width: float,
        slope_frac: float = 0.05) -> np.ndarray:
    """
    Taper the central part of a correlation matrix.

    Due to electromagnetic cross-talk, signal processing or other effects the
    correlaton matrices are often contaminated around the zero lag time. This
    function tapers (multiples by zero) the central part of width `width`. To
    avoid problems with interpolation and filtering later on this is done with
    cosine taper

    :param data: Correlation data from CorrBulk object
    :type data: np.ndarray
    :param stats: Stats object from CorrBulk
    :type stats: CorrStats
    :param width: width of the central window to be tapered in seconds
        (in total, i.e. not per taper side)
    :type width: float
    :param slope_frac: fraction of `width` used for soothing of edges,
        defaults to 0.05
    :type slope_frac: float, optional
    :return: The tapered matrix
    :rtype: np.ndarray
    """
    if width == 0:
        return data
    elif width < 0:
        raise ValueError('Taper Width must not be negative.')
    elif width > stats['npts']/stats['sampling_rate']:
        raise ValueError('Taper width longer than signal.')

    # calculate size of taper (should be an even number)
    length = int(2. * np.ceil(width * stats['sampling_rate'] / 2.))
    slope_length = int(np.ceil(length * slope_frac / 2.))

    # calculate inverse taper
    taper = np.zeros(length + 1)
    tap = cosine_taper(slope_length * 2, p=1)
    taper[0:slope_length] = tap[slope_length:]
    taper[-slope_length:] = tap[:slope_length]

    # calculate start end end points of the taper
    start = int(-stats['start_lag']*stats['sampling_rate'] - length/2)
    if start < 0:
        taper = taper[np.absolute(start):]
        start = 0

    end = start + length
    if end > stats['npts']:
        end = stats['npts']
        taper = taper[0:end - start]
    # apply taper
    data[:, start:end + 1] *= \
        np.tile(np.atleast_2d(taper), (data.shape[0], 1))

    return data


def corr_mat_resample_time(
    data: np.ndarray, stats: CorrStats, freq: float) -> Tuple[
        np.ndarray, CorrStats]:
    """
    Resample the lapse time axis of a correlation matrix. The correlations are
    automatically filtered with a highpass filter of 0.4*sampling frequency to
    avoid aliasing.

    :type data: np.ndarray
    :param data: correlation matrix
    :type stats: :class:`~seismic.correlate.stats.CorrStats`
    :param stats: The stats object associated to the
        :class:`~seismic.correlate.stream.CorrBulk` object.
    :type freq: float
    :param freq: new sampling frequency

    :rtype: Tuple[np.ndarray, CorrStats]
    :return: Resampled data and changed stats object.

    ..note:: This action is performed **in-place**.
    """
    if freq > stats['sampling_rate']:
        raise ValueError('New sampling frequency higher than original f.')
    elif freq == stats['sampling_rate']:
        return data, stats

    # make last samples
    # x1 = n2*f1/f2-n1 + x2*f1/f2
    otime = float(stats['npts'])/stats['sampling_rate']
    num = int(np.floor(otime * freq))
    ntime = num/freq
    n1 = 0.
    n2 = 0.

    while ntime != otime:
        if ntime < otime:
            n1 += 1
            ntime = (num + n1)/freq
        else:
            n2 += 1
            otime = float(stats['npts']+n2)/stats['sampling_rate']
        if n1 > num:
            raise TypeError("No matching trace length found. Resampling not \
possible.")

    data = corr_mat_filter(data, stats, [0.001, 0.8*freq/2])

    data = np.concatenate(
        (data, np.zeros((data.shape[0], data.shape[1] + int(2*n2)))), axis=1)
    data = resample(data, int(2*(num + n1)), axis=1)[:, :num]
    stats['npts'] = data.shape[1]
    stats['sampling_rate'] = freq
    return data, stats


def corr_mat_decimate(
    data: np.ndarray, stats: CorrStats, factor: int) -> Tuple[
        np.ndarray, CorrStats]:
    """
    Downsample a correlation matrix by an integer sample. A low-pass filter
    is applied before decimating.

    :type data: np.ndarray
    :param data: correlation matrix
    :type stats: :class:`~seismic.correlate.stats.CorrStats`
    :param stats: The stats object associated to the
        :class:`~seismic.correlate.stream.CorrBulk` object.
    :type factor: int
    :param factor: The factor to reduce the sampling frequency by

    :rtype: Tuple[np.ndarray, CorrStats]
    :return: Decimated data and changed stats object.

    ..note:: This action is performed **in-place**.
    """
    if factor < 1:
        raise ValueError('Factor has to be larger than 1')
    elif factor == 1:
        return data, stats
    # apply low pass filter
    freq = stats['sampling_rate'] * 0.5 / float(factor)

    fe = float(stats['sampling_rate']) / 2

    (b, a) = butter(4, freq / fe, btype='lowpass')

    # forward
    data = lfilter(b, a, data, axis=1)
    # backwards
    data = lfilter(b, a, data[:, ::-1], axis=1)[:, ::-factor][:, :-1]
    # For some reason it includes the last sample
    stats['npts'] = data.shape[1]
    stats['sampling_rate'] = float(stats['sampling_rate'])/factor
    return data, stats


def corr_mat_resample_or_decimate(
    data: np.ndarray, stats: CorrStats, freq: float) -> Tuple[
        np.ndarray, CorrStats]:
    """
    Downsample a correlation matrix. Decides automatically whether to decimate
    or downsampled depneding on the target frequency. A low-pass filter
    is applied to avoid aliasing.

    :type data: np.ndarray
    :param data: correlation matrix
    :type stats: :class:`~seismic.correlate.stats.CorrStats`
    :param stats: The stats object associated to the
        :class:`~seismic.correlate.stream.CorrBulk` object.
    :type freq: float
    :param freq: new sampling frequency

    :rtype: Tuple[np.ndarray, CorrStats]
    :return: Decimated data and changed stats object.

    ..note:: This action is performed **in-place**.
    """
    sr = stats['sampling_rate']

    if sr/freq == sr//freq:
        return corr_mat_decimate(data, stats, int(sr//freq))
    else:
        return corr_mat_resample_time(data, stats, freq)


def corr_mat_extract_trace(
    data: np.ndarray, stats: CorrStats, method: str = 'mean',
        percentile: float = 50.) -> np.ndarray:
    """ Extract a representative trace from a correlation matrix.

    Extract a correlation trace from the that best represents the correlation
    matrix. ``Method`` decides about method to extract the trace. The following
    possibilities are available

    * ``mean`` averages all traces in the matrix
    * ``norm_mean`` averages the traces normalized after normalizing for maxima
    * ``similarity_percentile`` averages the ``percentile`` % of traces that
        best correlate with the mean of all traces. This will exclude abnormal
        traces. ``percentile`` = 50 will return an average of traces with
        correlation (with mean trace) above the median.

    :type corr_mat: dictionary
    :param corr_mat: correlation matrix dictionary
    :type method: string
    :param method: method to extract the trace
    :type percentile: float
    :param percentile: only used for method=='similarity_percentile'

    :rtype: trace dictionary of type correlation trace
    :return **trace**: extracted trace
    """

    ndata = deepcopy(data)

    if method == 'mean':
        mm = np.ma.masked_array(
            ndata, np.isnan(ndata))
        out = np.mean(mm, 0).filled(np.nan)
    elif method == 'norm_mean':
        # normalize the matrix
        ndata = corr_mat_normalize(ndata, stats, normtype='absmax')
        mm = np.ma.masked_array(
            ndata, np.isnan(ndata))
        out = (np.mean(mm, 0).filled(np.nan))
    elif method == 'similarity_percentile':
        # normalize the matrix
        ndata = corr_mat_normalize(ndata, stats, normtype='absmax')
        mm = np.ma.masked_array(
            ndata, np.isnan(ndata))
        # calc mean  trace
        mean_tr = np.mean(mm, 0).filled(np.nan)
        # calc similarity with mean trace
        tm_sq = np.sum(mean_tr ** 2)
        cm_sq = np.sum(ndata ** 2, axis=1)
        # cm_sq = np.sum(data ** 2, axis=1).filled(np.nan)
        cor = np.zeros(ndata.shape[0])
        for ind in range(mm.shape[0]):
            cor[ind] = (
                np.dot(ndata[ind, :], mean_tr) / np.sqrt(cm_sq[ind] * tm_sq))
        # estimate the percentile excluding nans
        tres = np.percentile(cor[~np.isnan(cor)], percentile)
        # find the traces that agree with the requested percentile and calc
        # their mean
        ind = cor > tres
        out = np.mean(ndata[ind, :], 0)
    else:
        raise ValueError("Method '%s' not defined." % method)

    return out


class Error(Exception):
    pass


class InputError(Error):
    def __init__(self,  msg):
        self.msg = msg


def corr_mat_stretch(
    cdata: np.ndarray, stats: CorrStats, ref_trc: np.ndarray = None,
    tw: List[np.ndarray] = None, stretch_range: float = 0.1,
    stretch_steps: int = 100, sides: str = 'both',
        return_sim_mat: bool = False) -> dict:
    """ Time stretch estimate through stretch and comparison.

    This function estimates stretching of the time axis of traces as it can
    occur if the propagation velocity changes.

    Time stretching is estimated comparing each correlation function stored
    in the ``corr_data`` matrix (one for each row) with ``stretch_steps``
    stretched versions  of reference trace stored in ``ref_trc``. If
    ``ref_trc`` is ``None`` the mean of all traces is used.
    The maximum amount of stretching may be passed in ``stretch_range``. The
    time axis is multiplied by exp(stretch).
    The best match (stretching amount and corresponding correlation value) is
    calculated on different time windows. If ``tw = None`` the stretching is
    estimated on the whole trace.

    :type corr_mat: dictionary
    :param corr_mat: correlation matrix dictionary as produced by
        :class:`~miic.core.macro.recombine_corr_data`
    :type ref_trc: :class:`~numpy.ndarray`
    :param ref_trc: 1D array containing the reference trace to be stretched
        and compared to the individual traces in ``mat``
    :type tw: list of :class:`~numpy.ndarray` of int
    :param tw: list of 1D ndarrays holding the indices of samples in the time
        windows to be use in the time shift estimate. The samples are counted
        from the zero lag time with the index of the first sample being 0. If
        ``tw = None`` the full time range is used.
    :type stretch_range: scalar
    :param stretch_range: Maximum amount of relative stretching.
        Stretching and compression is tested from ``-stretch_range`` to
        ``stretch_range``.
    :type stretch_steps: scalar`
    :param stretch_steps: Number of shifted version to be tested. The
        increment will be ``(2 * stretch_range) / stretch_steps``
    :type sides: str
    :param sides: Side of the reference matrix to be used for the stretching
        estimate ('both' | 'left' | 'right' | 'single') ``single`` is used for
        one-sided signals from active sources with zero lag time is on the
        first sample. Other options assume that the zero lag time is in the
        center of the traces.


    :rtype: Dictionary
    :return: **dv**: Dictionary with the following keys

        *corr*: 2d ndarray containing the correlation value for the best
            match for each row of ``mat`` and for each time window.
            Its dimension is: :func:(len(tw),mat.shape[1])
        *value*: 2d ndarray containing the stretch amount corresponding to
            the best match for each row of ``mat`` and for each time window.
            Stretch is a relative value corresponding to the negative relative
            velocity change -dv/v.
            Its dimension is: :func:(len(tw),mat.shape[1])
        *sim_mat*: 3d ndarray containing the similarity matricies that
            indicate the correlation coefficient with the reference for the
            different time windows, different times and different amount of
            stretching.
            Its dimension is: :py:func:`(len(tw),mat.shape[1],len(strvec))`
        *second_axis*: It contains the stretch vector used for the velocity
            change estimate.
        *vale_type*: It is equal to 'stretch' and specify the content of
            the returned 'value'.
        *method*: It is equal to 'single_ref' and specify in which "way" the
            values have been obtained.
    """

    data = deepcopy(cdata)

    dta = -stats['start_lag']
    dte = stats['end_lag']

    # format (trimm) the matrix for zero-time to be either at the beginning
    # or at the center as required by
    # miic.core.stretch_mod.time_stretch_estimate

    # In case a reference is provided but the matrix needs to be trimmed the
    # references also need to be trimmed. To do so we append the references to
    # the matrix, trimm it and remove the references again
    if ref_trc is not None:
        rts = ref_trc.shape
        if len(rts) == 1:
            nr = 1
        else:
            nr = rts[0]
        data = np.concatenate((
            data, np.atleast_2d(ref_trc)), 0)
        reft = np.tile([UTCDateTime(1900, 1, 1)], (nr))
        stats['corr_start'] = np.concatenate((stats['corr_start'], reft), 0)

    # trim the matrices
    if sides == "single":
        # extract the time>0 part of the matrix
        data, stats = corr_mat_trim(data, stats, 0, dte)
    else:
        # extract the central symmetric part (if dt<0 the trim will fail)
        dt = min(dta, dte)
        data, stats = corr_mat_trim(data, stats, -dt, dt)

    # create or extract references
    if ref_trc is None:
        ref_trc = corr_mat_extract_trace(data, stats)
    else:
        # extract and remove references from corr matrix again
        ref_trc = data[-nr:, :]
        data = data[:-nr, :]
        stats['corr_start'] = stats['corr_start'][:-nr]

    dv = multi_ref_vchange_and_align(
        data, ref_trc, tw=tw, stretch_range=stretch_range,
        stretch_steps=stretch_steps, sides=sides,
        return_sim_mat=return_sim_mat)

    # add the keys the can directly be transferred from the correlation matrix
    dv['stats'] = stats

    return dv


def corr_mat_shift(
    data: np.ndarray, stats: CorrStats, ref_trc: np.ndarray = None,
    tw: List[np.ndarray] = None, shift_range: float = 10,
    shift_steps: int = 101, sides: str = 'both',
        return_sim_mat: bool = False) -> dict:
    """ Time shift estimate through shifting and comparison.

    This function estimates shifting of the time axis of traces as it can
    occur if the clocks of digitizers drift.

    Time shifts are estimated comparing each correlation function stored
    in the ``corr_data`` matrix (one for each row) with ``shift_steps``
    shifted versions  of reference trace stored in ``ref_trc``.
    The maximum amount of shifting may be passed in ``shift_range``.
    The best match (shifting amount and corresponding correlation value) is
    calculated on different time windows. If ``tw = None`` the shifting is
    estimated on the whole trace.

    :type corr_mat: dictionary
    :param corr_mat: correlation matrix dictionary as produced by
        :class:`~miic.core.macro.recombine_corr_data`
    :type ref_trc: :class:`~numpy.ndarray`
    :param ref_trc: 1D array containing the reference trace to be stretched
        and compared to the individual traces in ``mat``
    :type tw: list of :class:`~numpy.ndarray` of int
    :param tw: list of 1D ndarrays holding the indices of sampels in the time
        windows to be use in the time shift estimate. The sampels are counted
        from the zero lag time with the index of the first sample being 0. If
        ``tw = None`` the full time range is used.
    :type shift_range: scalar
    :param shift_range: Maximum amount of shifting in samples.
        Shifting is tested from ``-stretch_range`` to
        ``stretch_range``.
    :type shift_steps: scalar`
    :param shift_steps: Number of shifted version to be tested. The
        increment will be ``(2 * shift_range) / shift_steps``
    :type sides: str
    :param sides: Side of the reference matrix to be used for the shifting
        estimate ('both' | 'right') ``single`` is used for
        one-sided signals from active sources with zero lag time is on the
        first sample. Other options assume that the zero lag time is in the
        center of the traces.


    :rtype: Dictionary
    :return: **dv**: Dictionary with the following keys

        *corr*: 2d ndarray containing the correlation value for the best
            match for each row of ``mat`` and for each time window.
            Its dimension is: :func:(len(tw),mat.shape[1])
        *value*: 2d ndarray containing the stretch amount corresponding to
            the best match for each row of ``mat`` and for each time window.
            Stretch is a relative value corresponding to the negative relative
            velocity change -dv/v.
            Its dimension is: :func:(len(tw),mat.shape[1])
        *sim_mat*: 3d ndarray containing the similarity matricies that
            indicate the correlation coefficient with the reference for the
            different time windows, different times and different amount of
            stretching.
            Its dimension is: :py:func:`(len(tw),mat.shape[1],len(strvec))`
        *second_axis*: It contains the stretch vector used for the velocity
            change estimate.
        *vale_type*: It is equal to 'shift' and specify the content of
            the returned 'value'.
        *method*: It is equal to 'single_ref' and specify in which "way" the
            values have been obtained.
    """

    data = deepcopy(data)

    dta = -stats.start_lag
    dte = stats.end_lag

    # format (trimm) the matrix for zero-time to be either at the beginning
    # or at the center as required by
    # miic.core.stretch_mod.time_stretch_estimate

    # In case a reference is provided but the matrix needs to be trimmed the
    # reference also needs to be trimmed. To do so we append the reference to
    # the matrix, trimm it and remove the reference again
    if ref_trc is not None:
        rts = ref_trc.shape
        if len(rts) == 1:
            nr = 1
        else:
            nr = rts[0]
        data = np.concatenate((
            data, np.atleast_2d(ref_trc)), 0)
        reft = np.tile([UTCDateTime(1900, 1, 1)], (nr))
        stats['corr_start'] = np.concatenate((stats['corr_start'], reft), 0)

    # trim the marices
    if sides == "single":
        # extract the time>0 part of the matrix
        data, stats = corr_mat_trim(data, stats, 0, dte)
    else:
        # extract the central symmetric part (if dt<0 the trim will fail)
        dt = min(dta, dte)
        data, stats = corr_mat_trim(data, stats, -dt, dt)

    # create or extract references
    if ref_trc is None:
        ref_trc = corr_mat_extract_trace(data, stats)
    else:
        # extract and remove references from corr matrix again
        ref_trc = np.squeeze(data[-nr:, :])
        data = data[:-nr, :]
        stats['corr_start'] = stats['corr_start'][:-nr]

    # set sides
    if sides == 'both':
        ss = False
    elif sides == 'right':
        ss = True
    else:
        raise ValueError(
            "Error: side is not recognized. Use either both or right.")

    dt = time_shift_estimate(
        data, ref_trc=ref_trc, tw=tw, shift_range=shift_range,
        shift_steps=shift_steps, single_sided=ss,
        return_sim_mat=return_sim_mat)

    # add the keys the can directly be transferred from the correlation matrix
    dt['stats'] = stats
    return dt


def measure_shift(
    data: np.ndarray, stats: CorrStats, ref_trc: np.ndarray | None = None,
    tw: List[List] | None = None, shift_range: float = 10,
    shift_steps: int = 101, sides: str = 'both',
        return_sim_mat: bool = False) -> List[DV]:
    """ Time shift estimate through shifting and comparison.

    This function estimates shifting of the time axis of traces as it can
    occur if the clocks of digitizers drift.

    Time shifts are estimated comparing each trace (e.g. correlation function
    stored in the ``corr_data`` matrix (one for each row) with shifted
    versions  of reference trace stored in ``ref_trc``. The range of shifting
    to be tested is given in ``shift_range`` in seconds. It is used in a
    symmetric way from -``shift_range`` to +``shift_range``. Shifting ist
    tested ``shift_steps`` times. ``shift_steps`` should be an odd number to
    test zero shifting. The best match (shifting amount and corresponding
    correlation value) is calculated in specified time windows. Multiple time
    windows may be specified in  ``tw``.


    :type data: :class:`~numpy.ndarray`
    :param data: correlation matrix or collection of traces. lag time runs
        along rows and number of traces is ``data.shape[0]``.
    :type ref_trc: :class:`~numpy.ndarray`
    :param ref_trc: 1D array containing the reference trace to be shifted
        and compared to the individual traces in ``data``. If ``ref_trc``is
        None, the average of all traces in ``data`` is used.
    :type tw: list of lists with two floats
    :param tw: list of lists that contain the start and end times of the time
        windows in which the similarity is to be estimated. Time are given in
        seconds of lag time with respect to zero lag time. Using multiple
        lists of start end end time the best shift can be estimated in
        multiple time windows. If ``tw = None`` one time window is used
        containing the full time range of lag times in the traces.
    :type shift_range: scalar
    :param shift_range: Maximum amount of shifting in second.
        Shifting is tested from ``-stretch_range`` to
        ``stretch_range``.
    :type shift_steps: scalar`
    :param shift_steps: Number of shifted version to be tested. The
        increment will be ``(2 * shift_range) / shift_steps``
    :type sides: str
    :param sides: Side of the traces to be used for the shifting estimate
        ('both' | 'single'). ``single`` is used for
        one-sided signals from active sources or if the time window shall
        not be symmetric. For ``both`` the time window will be mirrowd about
        zero lag time, e.g. [start,end] will result in time windows
        [-end:-start] and [start:end] being used simultaneousy


    :rtype: List[Dictionary]
    :return: **dv**: List of ``len(tw)`` Dictionaries with the following keys

        *corr*: 2d ndarray containing the correlation value for the best
            match for each row of ``mat`` and for each time window.
            Its dimension is: :func:(len(tw),mat.shape[1])
        *value*: 2d ndarray containing the stretch amount corresponding to
            the best match for each row of ``mat`` and for each time window.
            Stretch is a relative value corresponding to the negative relative
            velocity change -dv/v.
            Its dimension is: :func:(len(tw),mat.shape[1])
        *sim_mat*: 3d ndarray containing the similarity matricies that
            indicate the correlation coefficient with the reference for the
            different time windows, different times and different amount of
            stretching.
            Its dimension is: :py:func:`(len(tw),mat.shape[1],len(strvec))`
        *second_axis*: It contains the stretch vector used for the velocity
            change estimate.
        *vale_type*: It is equal to 'shift' and specify the content of
            the returned 'value'.
        *method*: It is equal to 'single_ref' and specify in which "way" the
            values have been obtained.
    """

    if sides not in ['both', 'single']:
        raise ValueError(
            f"'sides' must be either 'both' or 'single', not {sides}")
    if tw is not None:
        for twi in tw:
            if not len(twi) == 2:
                raise ValueError(
                    f"individual time windows must be specified as two-item"
                    f"list, not {twi} of length {len(twi)}.")

    if sides == 'both':
        if np.any(np.array(tw) < 0):
            raise ValueError(
                "For 'sides=='both' all values in tw must be >= 0")
    if ref_trc is not None:
        if len(ref_trc) != data.shape[1]:
            raise ValueError(
                "Length of 'ref_trc' must match 'data.shape[1]', "
                f"they are {len(ref_trc)} and {data.shape[1]}.")

    data = deepcopy(data)
    stats = deepcopy(stats)

    # If reference trace is given attach it to the data for common
    # preprocessing
    if ref_trc is not None:
        data = np.concatenate((
            data, np.atleast_2d(ref_trc)), 0)
        reft = np.tile([UTCDateTime(1900, 1, 1)], (1))
        stats['corr_start'] = np.concatenate((stats['corr_start'], reft), 0)
    # If no time window is given use the whole range
    if tw is None:
        tw = [[stats.start_lag, stats.end_lag]]

    # trim to required lapse time range
    twmax = np.max(np.array(tw)[:]) + shift_range
    if sides == 'both':
        twmin = -twmax
    else:
        twmin = np.min(np.array(tw)[:]) - shift_range
    assert twmax <= stats.end_lag, "Time window ends later than the trace. " \
        f"Latest time window + shift_range ends at {twmax}s " \
        f"while trace end at {stats.end_lag}s."
    assert twmin >= stats.start_lag, f"Time window starts earlier than the " \
        f"trace. Earliest time window - shift_range starts at {twmin}s " \
        f"while trace starts at {stats.start_lag}s."
    data, stats = corr_mat_trim(data, stats, twmin, twmax)

    # create or extract references
    if ref_trc is None:
        ref_trc = corr_mat_extract_trace(data, stats)
    else:
        # extract and remove references from corr matrix again
        ref_trc = np.squeeze(data[-1:, :])
        data = data[:-1, :]
        stats['corr_start'] = stats['corr_start'][:-1]

    # create matrix with shifted references
    shifts = np.linspace(-shift_range, shift_range, shift_steps)
    ref_mat = create_shifted_ref_mat(ref_trc, stats, shifts)

    # create space for results
    dt_list = []
    # do computation for each time window
    for twi in tw:
        # create indices of time windows
        indices = np.arange(
            np.ceil(twi[0]*stats.sampling_rate),
            np.floor(twi[1]*stats.sampling_rate))
        if sides == 'both':
            indices = np.concatenate((np.flipud(-indices), indices))
        indices -= np.round(stats.start_lag * stats.sampling_rate)
        indices = indices.astype(int)
        # compare data and shifted reference
        sim_mat = compare_with_modified_reference(data, ref_mat, indices)
        corr = sim_mat.max(axis=1)
        value = shifts[sim_mat.argmax(axis=1)]
        # Set dt to NaN where the correlation is NaN instead of having it equal
        # to one of the two stretch_range limits
        value[np.isnan(corr)] = np.nan
        # assemble results
        dt = {
            'stats': stats,
            'corr': np.squeeze(corr),
            'value': np.squeeze(value),
            'second_axis': shifts,
            'value_type': np.array(['shift']),
            'method': np.array(['absolute_shift'])}
        if return_sim_mat:
            dt.update({'sim_mat': np.squeeze(sim_mat)})
        else:
            dt.update({'sim_mat': None})

        dt_list.append(DV(**dt))

    return dt_list


def apply_shift(
    data: np.ndarray, stats: CorrStats, shifts: np.ndarray) -> Tuple[
        np.ndarray, CorrStats]:
    """ Apply a shift to a correlation matrix.

    This function applies a shift given in seconds to each trace in the matrix.

    :type data: :class:`~numpy.ndarray`
    :param data: correlation matrix or collection of traces. lag time runs
        along rows and number of traces is ``data.shape[0]``.
    :type shifts: :class:`~numpy.ndarray`
    :param shifts: shifts in seconds for each trace in 'data'
    """
    data = deepcopy(data)
    times = np.linspace(stats.start_lag, stats.end_lag, stats.npts)

    # stretch every line
    for (ii, line) in enumerate(data):
        s = UnivariateSpline(times, line, s=0)
        data[ii, :] = s(times+shifts[ii])

    return data, stats


def apply_stretch(
    data: np.ndarray, stats: CorrStats, stretches: np.ndarray) -> Tuple[
        np.ndarray, CorrStats]:
    """ Apply a stretches to a correlation matrix.

    This function applies a stretch given in relative units to each trace
    in the matrix.

    :type data: :class:`~numpy.ndarray`
    :param data: correlation matrix or collection of traces. lag time runs
        along rows and number of traces is ``data.shape[0]``.
    :type stretches: :class:`~numpy.ndarray`
    :param stretches: stretches in relative units for each trace in 'data'
    """
    data = deepcopy(data)
    times = np.linspace(stats.start_lag, stats.end_lag, stats.npts)

    # stretch every line
    for (ii, line) in enumerate(data):
        s = UnivariateSpline(times, line, s=0)
        data[ii, :] = s(times * np.exp(-stretches[ii]))

    return data, stats
