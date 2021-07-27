'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Christoph Sens-Schönefelder
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 14th June 2021 08:50:57 am
Last Modified: Tuesday, 27th July 2021 10:53:28 am
'''

from typing import List, Tuple
import numpy as np
from copy import deepcopy
from scipy.signal import butter, lfilter, hilbert, resample
from obspy.core import UTCDateTime

# Obspy imports
from obspy.signal.invsim import cosine_taper

from seismic.monitor.stretch_mod import multi_ref_vchange_and_align, \
    time_shift_estimate
from seismic.correlate.stats import CorrStats


# This needs some tidying


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
    :param starttime: start time in seconds with respect to the zero position
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
    data = data[:, start: end + 1]

    # set starttime, endtime and npts of the new stats
    stats['start_lag'] = starttime
    # This should change automatically
    # stats['end_lag'] = endtime
    # Not really in use, so why bothering
    # stats['starttime'] = lag0 + starttime
    stats['npts'] = data.shape[1]
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
    otime = [ii.timestamp for ii in stats['corr_start']]
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
            etime = stats['corr_end'][-1]
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


# def corr_mat_reverse(corr_mat):
#     """ Reverse the data in a correlation matrix.

#     If a correlation matrix was calculated for a pair of stations sta1-sta2
#     this function reverses the causal and acausal parts of the correlation
#     matrix as if the correlations were calculated for the pair sta2-sta1.

#     :type corr_mat: dictionary
#     :param corr_mat: correlation matrix dictionary as produced by
#         :class:`~miic.core.macro.recombine_corr_data`

#     :rtype: dictionary
#     :return: **corr_mat**: is the same dictionary as the input but with
#         reversed order of the station pair.
#     """

#     # check input
#     if not isinstance(corr_mat, dict):
#         raise TypeError("corr_mat needs to be correlation matrix dictionary.")

#     if corr_mat_check(corr_mat)['is_incomplete']:
#         raise ValueError("Error: corr_mat is not a valid correlation_matix \
#             dictionary.")

#     # check input
#     zerotime = lag0

#     # reverse the stats_tr1 and stats_tr2
#     stats_tr1 = corr_mat['stats_tr1']
#     corr_mat['stats_tr1'] = corr_mat['stats_tr2']
#     corr_mat['stats_tr2'] = stats_tr1

#     # reverse the locations in the stats
#     stla = corr_mat['stats']['stla']
#     stlo = corr_mat['stats']['stlo']
#     stel = corr_mat['stats']['stel']
#     corr_mat['stats']['stla'] = corr_mat['stats']['evla']
#     corr_mat['stats']['stlo'] = corr_mat['stats']['evlo']
#     corr_mat['stats']['stel'] = corr_mat['stats']['evel']
#     corr_mat['stats']['evla'] = stla
#     corr_mat['stats']['evlo'] = stlo
#     corr_mat['stats']['evel'] = stel

#     # reverse azimuth and backazimuth
#     tmp = corr_mat['stats']['az']
#     corr_mat['stats']['az'] = corr_mat['stats']['baz']
#     corr_mat['stats']['baz'] = tmp

#     # reverse the matrix
#     corr_mat['corr_data'] = corr_mat['corr_data'][:, -1::-1]

#     # adopt the starttime and endtime
#     trace_length = timedelta(seconds=float(corr_mat['stats']['npts'] - 1) / \
#                              corr_mat['stats']['sampling_rate'])
#     endtime = convert_time([corr_mat['stats']['starttime']])[0] + trace_length
#     starttime = zerotime - (endtime - zerotime)
#     corr_mat['stats']['starttime'] = convert_time_to_string([starttime])[0]
#     corr_mat['stats']['endtime'] = \
#         convert_time_to_string([starttime + trace_length])[0]

#     # reverse the seedID
#     keywords = ['network', 'station', 'location', 'channel']
#     for key in keywords:
#         if key in corr_mat['stats_tr1'] and key in corr_mat['stats_tr2']:
#             # empty loctions show up as empty lists -> convert in empty string
#             if isinstance(corr_mat['stats_tr1'][key], list):
#                 corr_mat['stats_tr1'][key] = ''
#             if isinstance(corr_mat['stats_tr2'][key], list):
#                 corr_mat['stats_tr2'][key] = ''
#             corr_mat['stats'][key] = \
#                 corr_mat['stats_tr1'][key] + '-' + corr_mat['stats_tr2'][key]

#     return corr_mat


# def corr_mat_time_select(corr_mat, starttime=None, endtime=None):
#     """ Select time period from a correlation matrix.

#     Select correlation  traces from a correlation matrix that fall into the
#     time period `starttime`<= selected times <= `endtime` and return them in
#     a correlation matrix.

#     :type corr_mat: dictionary
#     :param corr_mat: correlation matrix dictionary as produced by
#         :class:`~miic.core.macro.recombine_corr_data`
#     :type starttime: datetime.datetime object or time string
#     :param starttime: beginning of the selected time period
#     :type endtime: datetime.datetime object or time string
#     :param endtime: end of the selected time period

#     :rtype: dictionary
#     :return: **corr_mat**: correlation matrix dictionary restricted to the
#         selected time period.
#     """

#     # check input
#     if not isinstance(corr_mat, dict):
#         raise TypeError("corr_mat needs to be correlation matrix dictionary.")

#     if corr_mat_check(corr_mat)['is_incomplete']:
#         raise ValueError("Error: corr_mat is not a valid correlation_matix \
#             dictionary.")

#     smat = deepcopy(corr_mat)

#     # convert time vector
#     time = convert_time(corr_mat['time'])

#     # convert starttime and endtime input.
#     # if they are None take the first or last values of the time vector
#     if starttime == None:
#         starttime = time[0]
#     else:
#         if not isinstance(starttime, datetime):
#             starttime = convert_time([starttime])[0]
#     if endtime == None:
#         endtime = time[-1]
#     else:
#         if not isinstance(endtime, datetime):
#             endtime = convert_time([endtime])[0]

#     # select period
#     ind = np.nonzero((time >= starttime) * (time < endtime))[0]  # ind is
#                                                 #  a list(tuple) for dimensions

#     # trim the matrix
#     smat['corr_data'] = corr_mat['corr_data'][ind, :]

#     # adopt time vector
#     smat['time'] = corr_mat['time'][ind]
#     #smat['time'] = np.take(corr_mat['time'],ind)

#     return smat


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
    # cs = (zerotime - convert_time([corr_mat['stats']['starttime']])[0]). \
    #         total_seconds() * corr_mat['stats']['sampling_rate']
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

    # if zero_sample <= 0:
    #     print('No data present for mirrowing: starttime > zerotime.')
    #     return corr_mat
    # if convert_time([corr_mat['stats']['endtime']])[0] <= zerotime:
    #     print('No data present for mirrowing: endtime < zerotime.')
    #     return corr_mat
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
    # acausal_samples = int((zerotime -
    #         convert_time([corr_mat['stats']['starttime']])[0]). \
    #         total_seconds() * corr_mat['stats']['sampling_rate'] + 1)
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


# def corr_trace_prep_aftan(corr_trace,outname) :
#     """ Prepare and write out a SAC file for analysis in aFTAN
#     The b , e and dist headers are set in the sac header

#     :type corr_trace: dictionary
#     :param corr_trace: correlation trace dictionary as produced by
#         :class:`~miic.core.corr_mat_processing.corr_mat_extract_trace`
#     :type outname: string
#     :param outname: string for outputfilename
#     """
#     trace = Stream()
#     trace.append(corr_trace_to_obspy(corr_trace))
#     sacstats=AttribDict({'b':-(trace[0].stats.npts-1)/2,'e':(trace[0].stats.npts-1)/2,
#                         'dist':corr_trace['stats']['dist']})
#     trace[0].stats.sac=sacstats
#     trace.write(outname,format="SAC")
#     return


# def corr_trace_to_avg_sac(corr_trace,sacfname) :
#     """ Write a SAC file for FTAN analysis on a
#     one-side correlation trace. 
#     :type corr_trace: dictionary
#     :param corr_trace: correlation trace dictionary as produced by
#         :class:`~miic.core.corr_mat_processing.corr_mat_extract_trace`
#     :type outname: string
#     :param outname: string for outputfilename
#     """
#     avg_corr_trace = corr_trace_mirrow(corr_trace)
#     trace = Stream()
#     trace.append(corr_trace_to_obspy(avg_corr_trace))
#     sacstats=AttribDict({'dist':corr_trace['stats']['dist']})
#     trace[0].stats.sac=sacstats
#     trace.write(sacfname,format="SAC")
#     return

# def corr_trace_to_full_sac(corr_trace,sacfname) :
#     """ Write a SAC file for FTAN analysis on a
#     two-sided correlation trace. 
#     :type corr_trace: dictionary
#     :param corr_trace: correlation trace dictionary as produced by
#         :class:`~miic.core.corr_mat_processing.corr_mat_extract_trace`
#     :type outname: string
#     :param outname: string for outputfilename
#     """
#     trace = Stream()
#     trace.append(corr_trace_to_obspy(corr_trace))
#     sacstats=AttribDict({'dist':corr_trace['stats']['dist']})
#     trace[0].stats.sac=sacstats
#     trace.write(sacfname,format="SAC")
#     return

# def corr_trace_mirrow(corr_tr):
#     """ Average the causal and acausal parts of a correlation trace.

#     :type corr_tr: dictionary
#     :param corr_tr: correlation trace dictionary as produced by
#         :class:`~miic.core.corr_mat_processing.corr_mat_extract_trace`

#     :rtype: dictionary
#     :return: **corr_tr**: is the same dictionary as the input but with
#         averaged causal and acausal parts of the correlation data.
#     """

#     zerotime = lag0

#     # copy input
#     mir_tr = deepcopy(corr_tr)

#     # check whether there is a sample at the zerotime
#     zero_sample = (zerotime -
#             convert_time([corr_tr['stats']['starttime']])[0]). \
#             total_seconds() * corr_tr['stats']['sampling_rate']
#     if zero_sample <= 0:
#         print('No data present for mirrowing: starttime > zerotime.')
#         return corr_tr
#     if convert_time([corr_tr['stats']['endtime']])[0] <= zerotime:
#         print('No data present for mirrowing: endtime < zerotime.')
#         return corr_tr
#     if np.fmod(zero_sample, 1) != 0:
#         print('need to shift for mirrowing')
#         return 0

#     # estimate size of mirrowed array
#     acausal_samples = int((zerotime -
#             convert_time([corr_tr['stats']['starttime']])[0]). \
#             total_seconds() * corr_tr['stats']['sampling_rate'] + 1)
#     causal_samples = int(corr_tr['stats']['npts'] - acausal_samples + 1)
#     # +1 because sample a zerotime counts twice
#     size = np.max([acausal_samples, causal_samples])
#     both = np.min([acausal_samples, causal_samples])

#     # allocate array
#     mir_tr['corr_trace'] = np.zeros(size)

#     # fill the array
#     mir_tr['corr_trace'][0:causal_samples] = \
#         corr_tr['corr_trace'][acausal_samples - 1:]
#     mir_tr['corr_trace'][0:acausal_samples] += \
#         corr_tr['corr_trace'][acausal_samples - 1::-1]

#     # divide by two where both are present
#     mir_tr['corr_trace'][0:both] /= 2.

#     # adopt the stats
#     mir_tr['stats']['starttime'] = convert_time_to_string([zerotime])[0]
#     mir_tr['stats']['npts'] = size
#     mir_tr['stats']['endtime'] = convert_time_to_string([zerotime +
#             timedelta(seconds=float(size) /
#             corr_tr['stats']['sampling_rate'])])[0]

#     return mir_tr


# def corr_trace_mirror_one_side(corr_tr,method) :
#     """ Mirror the causal or acausal parts of a correlation trace
#         to the other side

#     :type corr_tr: dictionary
#     :param corr_tr: correlation trace dictionary as produced by
#         :class:`~miic.core.corr_mat_processing.corr_mat_extract_trace`

#     :rtype: dictionary
#     :return: **corr_tr**: is the same dictionary as the input but with
#         averaged causal and acausal parts of the correlation data.
#     """

#     zerotime = lag0

#     # copy input
#     mir_tr = deepcopy(corr_tr)

#     # check whether there is a sample at the zerotime
#     zero_sample = (zerotime -
#             convert_time([corr_tr['stats']['starttime']])[0]). \
#             total_seconds() * corr_tr['stats']['sampling_rate']
#     if zero_sample <= 0:
#         print('No data present for mirrowing: starttime > zerotime.')
#         return corr_tr
#     if convert_time([corr_tr['stats']['endtime']])[0] <= zerotime:
#         print('No data present for mirrowing: endtime < zerotime.')
#         return corr_tr
#     if np.fmod(zero_sample, 1) != 0:
#         print('need to shift for mirrowing')
#         return 0

#     # estimate size of mirrowed array
#     acausal_samples = int((zerotime -
#             convert_time([corr_tr['stats']['starttime']])[0]). \
#             total_seconds() * corr_tr['stats']['sampling_rate'] + 1)
#     causal_samples = int(corr_tr['stats']['npts'] - acausal_samples + 1)
#     # +1 because sample a zerotime counts twice
#     size = np.max([acausal_samples, causal_samples])
#     both = np.min([acausal_samples, causal_samples])

#     # Fill array, mirroring one side to the other
#     mir_tr['corr_trace'] = np.zeros(corr_tr['corr_trace'].size)
#     if method=='causal' :
#         mir_tr['corr_trace'][causal_samples-1:]=corr_tr['corr_trace'][causal_samples-1:]
#         mir_tr['corr_trace'][:causal_samples]=corr_tr['corr_trace'][causal_samples-1:][::-1]
#     elif method=='acausal' :
#         mir_tr['corr_trace'][:acausal_samples]=corr_tr['corr_trace'][:acausal_samples]
#         mir_tr['corr_trace'][acausal_samples-1:]=corr_tr['corr_trace'][:acausal_samples][::-1]
#     return mir_tr


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
    # start = (zerotime - \
    #     convert_time([tmat['stats']['starttime']])[0]).total_seconds() * \
    #         tmat['stats']['sampling_rate'] - length / 2
    start = int(-stats['start_lag']*stats['sampling_rate'] - length/2)
    if start < 0:
        taper = taper[np.absolute(start):]
        start = 0
    # end = (zerotime - \
    #        convert_time([tmat['stats']['starttime']])[0]).total_seconds() * \
    #             tmat['stats']['sampling_rate'] + length / 2
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
            break
    data = corr_mat_filter(data, stats, [0.001, 0.8*freq/2])
    # rmat = {'stats':deepcopy(corr_mat['stats']),
    #         'stats_tr1':deepcopy(corr_mat['stats_tr1']),
    #         'stats_tr2':deepcopy(corr_mat['stats_tr2']),
    #         'time':deepcopy(corr_mat['time'])}
    data = np.concatenate(
        (data, np.zeros((data.shape[0], data.shape[1] + int(2*n2)))), axis=1)
    data = resample(data, int(2*(num + n1)), axis=1)[:,:num]
    # rmat.update({'corr_data':deepcopy(trmat[:,:num])})
    stats['npts'] = data.shape[1]
    stats['sampling_rate'] = freq
    # stats['end_lag'] = stats['start_lag'] + (stats['npts']-1)/freq
    # rmat['stats']['npts'] = rmat['corr_data'].shape[1]
    # rmat['stats']['sampling_rate'] = freq
    # endtime = convert_time([rmat['stats']['starttime']])[0]+timedelta(seconds=float(rmat['stats']['npts']-1)/freq)
    # rmat['stats']['endtime'] = convert_time_to_string([endtime])[0]
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

    # fdat = deepcopy(corr_mat)
    # fdat['corr_data'] = lfilter(b, a, fdat['corr_data'], axis=1)
    # fdat['corr_data'] = lfilter(b, a, fdat['corr_data'][:, ::-1],
    #                                     axis=1)[:,::-1]
    # forward
    data = lfilter(b, a, data, axis=1)
    # backwards
    data = lfilter(b, a, data[:, ::-1], axis=1)[:, ::-factor][:, :-1]
    # For some reason it includes the last sample
    stats['npts'] = data.shape[1]
    stats['sampling_rate'] = float(stats['sampling_rate'])/factor
    # stats['end_lag'] = stats['start_lag'] + (stats['npts']-1)/freq

    # fdat['corr_data'] = deepcopy(fdat['corr_data'][:,::factor])
    # fdat['stats']['npts'] = fdat['corr_data'].shape[1]
    # fdat['stats']['sampling_rate'] = float(fdat['stats']['sampling_rate'])/factor
    # start = convert_time([fdat['stats']['starttime']])[0]
    # length = (fdat['stats']['npts'] - 1)/fdat['stats']['sampling_rate']
    # fdat['stats']['endtime'] = convert_time_to_string([convert_time([fdat['stats']['starttime']])[0] \
    #         + timedelta(seconds=(fdat['stats']['npts'] - 1)/fdat['stats']['sampling_rate'])])[0]
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


# def corr_mat_import_US_stream(st, pretrigger=0):
#     """ Convert a stream with multiple traces from an active measurement
#     in a correlation matrix structure.

#     This functions takes a stream with multiple traces and puts them
#     in a correlation_matrix_structure to facilitate monitoring. It is
#     assumed that:
#     - traces are recorded by the same sensor
#     - signals are emitted from the same source
#     - stats['starttime'] is the time of the measurement
#     - the source time starts ``pretrigger`` seconds before starttime
#     - traces have the same length.
#     Metainformation are taken from st[0] only.

#     :type st: :class:`~obspy.core.stream.Stream`
#     :param st: data stream
#     :type pretrigger: float
#     :param pretrigger: time of source excitation in seconds after
#         st[0].stats['starttime']

#     :rtype: correlation matrix dictionary
#     :return: **corr_mat**: is the same dictionary as the input but with
#         normalized ampitudes.
#     """

#     zerotime = lag0
#     pretrig = timedelta(seconds=pretrigger)
#     tr = st[0]

#     dt = timedelta(seconds=1. / tr.stats['sampling_rate'])
#     corr_mat = {}
#     corr_mat['corr_data'] = np.zeros((len(st), st[0].stats['npts']))
#     time = []
#     if hasattr(tr, 'stats'):

#         _stats = {'network': tr.stats.network,
#                       'station': tr.stats.station,
#                       'location': tr.stats.location,
#                       'channel': tr.stats.channel,
#                       'sampling_rate': tr.stats.sampling_rate,
#                       'starttime': '%s' % tr.stats.starttime,
#                       'endtime': '%s' % tr.stats.endtime,
#                       'npts': tr.stats.npts}
#         _stats_tr1 = deepcopy(_stats)
#         _stats_tr2 = deepcopy(_stats)
#         _stats_tr1['network'] = 'SRC'
#         _stats_tr1['station'] = 'SRC'

#         if 'sac' in tr.stats:
#             _stats['stla'] = tr.stats.sac.stla
#             _stats['stlo'] = tr.stats.sac.stlo
#             _stats['stel'] = tr.stats.sac.stel
#             if np.all([x in tr.stats.sac for x in ['evla', 'evlo', 'evel', 'az', 'baz', 'dist']]):
#                 _stats['evla'] = tr.stats.sac.evla
#                 _stats['evlo'] = tr.stats.sac.evlo
#                 _stats['evel'] = tr.stats.sac.evel
#                 _stats['az'] = tr.stats.sac.az
#                 _stats['baz'] = tr.stats.sac.baz
#                 _stats['dist'] = tr.stats.sac.dist

#         corr_mat['stats'] = _stats
#         corr_mat['stats']['starttime'] = convert_time_to_string([zerotime -
#                 pretrig])[0]
#         corr_mat['stats']['endtime'] = convert_time_to_string([(zerotime -
#                 pretrig) + (tr.stats['npts'] - 1) * dt])[0]

#     for ind, tr in enumerate(st):
#         corr_mat['corr_data'][ind, :] = tr.data
#         time.append('%s' % tr.stats.starttime)

#     corr_mat['stats_tr1'] = corr_mat['stats']
#     corr_mat['stats_tr2'] = corr_mat['stats']

#     corr_mat['time'] = np.array(time)

#     return corr_mat


# def corr_mat_from_stream(st):
#     """ Create a correlation matrix from repeating source observations.

#     Given an obspy stream that contains traces of repeating events
#     it combines the traces in a corr_mat dictionary.

#     :type st: obspy.core.stream
#     :param st: stream that contains the data

#     :rtype: dictionary of type correlation matrix
#     :return: **corr_mat**

#     """

#     ID = st[0].stats['network']+'.'+st[0].stats['station']+'.'+st[0].stats['location']+'.'+st[0].stats['channel']
#     sampling_rate = st[0].stats['sampling_rate']
#     corr_mat = {'stats':_stats_dict_from_obj(st[0].stats),
#                 'stats_tr1':_stats_dict_from_obj(st[0].stats),
#                 'stats_tr2':_stats_dict_from_obj(st[0].stats),
#                 'corr_data':np.zeros((len(st),len(st[0].data))),
#                 'time':np.zeros(len(st),dtype=datetime)}

#     length = np.min([tr.stats.npts for tr in st])
#     # make it an odd number
#     length = int(2*np.floor(length/2) + 1)
#     starttime = zerotime
#     corr_mat['stats']['starttime'] = str(starttime)
#     corr_mat['stats']['npts'] = length
#     corr_mat['stats']['endtime'] = str(starttime + 1./sampling_rate*(length-1))
#     for ind,tr in enumerate(st):
#         corr_mat['time'][ind] = '%s' % tr.stats['starttime']
#         tID = tr.stats['network']+'.'+tr.stats['station']+'.'+tr.stats['location']+'.'+tr.stats['channel']
#         if tID != ID:
#             print("ID %s of trace %d does not match the first trace %s." % (tID, ind, ID))
#             continue
#         tsampling_rate = tr.stats['sampling_rate']
#         if tsampling_rate != sampling_rate:
#             print("Sampling rate of trace %d does not match the first trace %s." % (ind, sampling_rate))
#             continue

#         corr_mat['corr_data'][ind,:] = tr.data[:length]

#     return corr_mat


# def corr_mat_from_corr_stream(st):
#     """ Create a correlation matrix from an obspy stream.

#     Given an obspy stream that contains traces with the same geometrical
#     properties (i.e. location of source and receiver) usually obtained from
#     repeated measurements or seismic noise it combines the traces in a 
#     corr_mat dictionary.

#     :type st: obspy.core.stream
#     :param st: stream that contains the data

#     :rtype: dictionary of type correlation matrix
#     :return: **corr_mat**
    
#     """
    
#     ID = st[0].stats['network']+'.'+st[0].stats['station']+'.'+st[0].stats['location']+'.'+st[0].stats['channel']
#     starttime = st[0].stats['starttime']
#     sampling_rate = st[0].stats['sampling_rate']
#     npts = st[0].stats['npts']
#     corr_mat = {'stats':_stats_dict_from_obj(st[0].stats),
#                 'stats_tr1':_stats_dict_from_obj(st[0].stats_tr1),
#                 'stats_tr2':_stats_dict_from_obj(st[0].stats_tr2),
#                 'corr_data':np.zeros((len(st),len(st[0].data))),
#                 'time':np.zeros(len(st),dtype=datetime)}
#     for ind,tr in enumerate(st):
#         corr_mat['time'][ind] = '%s' % tr.stats_tr1['starttime']
#         tID = tr.stats['network']+'.'+tr.stats['station']+'.'+tr.stats['location']+'.'+tr.stats['channel']
#         if tID != ID:
#             print("ID %s of trace %d does not match the first trace %s." % (tID, ind, ID))
#             continue
#         tstarttime = tr.stats['starttime']
#         if tstarttime != starttime:
#             print("Starttime %s of trace %d does not match the first trace %s." % (tstarttime, ind, starttime))
#             continue
#         tsampling_rate = tr.stats['sampling_rate']
#         if tsampling_rate != sampling_rate:
#             print("Sampling rate of trace %d does not match the first trace %s." % (ind, sampling_rate))
#             continue
        
#         corr_mat['corr_data'][ind,:] = tr.data
    
#     return corr_mat


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
        

def corr_mat_rotate(corr_mat_list):
    """ Rotate correlation matrices from the NEZ in the RTZ frame. Corr_mat_list
    contains the NN,NE,NZ,EN,EE,EZ,ZN,ZE,ZZ components of the Greens tensor. Time
    vectors of all the correlation matricies must be identical.
    """
    
    cmd = {}
    required_comp = ['NN','NE','NZ','EN','EE','EZ','ZN','ZE','ZZ']
    times = corr_mat_list[0]['time']
    
    # quick fix the empty locations
    for cm in corr_mat_list:
        if cm['stats_tr1']['location'] == []: cm['stats_tr1']['location'] = ''
        if cm['stats_tr2']['location'] == []: cm['stats_tr2']['location'] = ''
        if cm['stats']['location'] == []: cm['stats']['location'] = ''
    
    # check IDs
    ID1 = (corr_mat_list[0]['stats_tr1']['network']+'.'+
           corr_mat_list[0]['stats_tr1']['station']+'.'+
           corr_mat_list[0]['stats_tr1']['location']+'.'+
           corr_mat_list[0]['stats_tr1']['channel'][:-1])
    ID2 = (corr_mat_list[0]['stats_tr2']['network']+'.'+
           corr_mat_list[0]['stats_tr2']['station']+'.'+
           corr_mat_list[0]['stats_tr2']['location']+'.'+
           corr_mat_list[0]['stats_tr2']['channel'][:-1])
    if ID1 == ID2:
        raise InputError('IDs of stations in corr_mat_list[0] may not be identical.')
        
    for ind,cm in enumerate(corr_mat_list):
        # check times match within 30 seconds
        if (cm['time'] != times).any():
            tol=30.0
            for arg in np.where(cm['time'] != times) :
                diff=np.abs(UTCDateTime(cm['time'][arg[0]])-UTCDateTime(times[arg[0]]))
                if diff > tol :
                    print(('Time vector mismatch greater than %ss for %s-%s %s %s' % \
                        (str(tol),cm['stats_tr1']['station'],cm['stats_tr2']['station'], \
                                cm['time'][arg[0]],times[arg[0]])))
                    #raise InputError('Time vector of corr_mat_list[%d] does not match the first matrix.' % ind)
        # check station IDs
        if (cm['stats_tr1']['network']+'.'+
            cm['stats_tr1']['station']+'.'+
            cm['stats_tr1']['location']+'.'+
            cm['stats_tr1']['channel'][:-1]) == ID1:
            if (cm['stats_tr2']['network']+'.'+
                cm['stats_tr2']['station']+'.'+
                cm['stats_tr2']['location']+'.'+
                cm['stats_tr2']['channel'][:-1]) != ID2:
                raise InputError('IDs of stations in corr_mat_list[%d] do not match IDs of'
                                 'corr_mat_list[0].' % ind)
        if (cm['stats_tr1']['network']+'.'+
            cm['stats_tr1']['station']+'.'+
            cm['stats_tr1']['location']+'.'+
            cm['stats_tr1']['channel'][:-1]) == ID2:
            if (cm['stats_tr2']['network']+'.'+
                cm['stats_tr2']['station']+'.'+
                cm['stats_tr2']['location']+'.'+
                cm['stats_tr2']['channel'][:-1]) != ID1:
                raise InputError('IDs of stations in corr_mat_list[%d] do not match IDs of'
                                 'corr_mat_list[0].' % ind)
                # flip corrmat
                cm = corr_mat_reverse(cm)
        
        cmd.update({cm['stats_tr1']['channel'][-1]+cm['stats_tr2']['channel'][-1]:cm})
    
    # check wheather all Greenstensor components are present
    for rc in required_comp:
        if rc not in list(cmd.keys()):
            print(rc, 'not present')
            #raise InputError('%s component is missing in corr_mat_list' % rc)

    
    return _rotate_corr_dict(cmd)
    


def corr_mat_rotate_horiz(corr_mat_list):
    """ Rotate correlation matrices from the NE into the RT frame. Corr_mat_list
    contains the NN,NE,EN,EE components of the Greens tensor. Time
    vectors of all the correlation matricies must be identical.
    """
    
    cmd = {}
    required_comp = ['NN','NE','EN','EE']
    times = corr_mat_list[0]['time']
    
    # quick fix the empty locations
    for cm in corr_mat_list:
        if cm['stats_tr1']['location'] == []: cm['stats_tr1']['location'] = ''
        if cm['stats_tr2']['location'] == []: cm['stats_tr2']['location'] = ''
        if cm['stats']['location'] == []: cm['stats']['location'] = ''
    
    # check IDs
    ID1 = (corr_mat_list[0]['stats_tr1']['network']+'.'+
           corr_mat_list[0]['stats_tr1']['station']+'.'+
           corr_mat_list[0]['stats_tr1']['location']+'.'+
           corr_mat_list[0]['stats_tr1']['channel'][:-1])
    ID2 = (corr_mat_list[0]['stats_tr2']['network']+'.'+
           corr_mat_list[0]['stats_tr2']['station']+'.'+
           corr_mat_list[0]['stats_tr2']['location']+'.'+
           corr_mat_list[0]['stats_tr2']['channel'][:-1])
    if ID1 == ID2:
        raise InputError('IDs of stations in corr_mat_list[0] may not be identical.')
        
    for ind,cm in enumerate(corr_mat_list):
        # check times match within 30 seconds
        if (cm['time'] != times).any():
            tol=30.0
            for arg in np.where(cm['time'] != times) :
                diff=np.abs(UTCDateTime(cm['time'][arg[0]])-UTCDateTime(times[arg[0]]))
                if diff > tol :
                    print(('Time vector mismatch greater than %ss for %s-%s %s %s' % \
                        (str(tol),cm['stats_tr1']['station'],cm['stats_tr2']['station'], \
                                cm['time'][arg[0]],times[arg[0]])))
                    #raise InputError('Time vector of corr_mat_list[%d] does not match the first matrix.' % ind)
        # check station IDs
        if (cm['stats_tr1']['network']+'.'+
            cm['stats_tr1']['station']+'.'+
            cm['stats_tr1']['location']+'.'+
            cm['stats_tr1']['channel'][:-1]) == ID1:
            if (cm['stats_tr2']['network']+'.'+
                cm['stats_tr2']['station']+'.'+
                cm['stats_tr2']['location']+'.'+
                cm['stats_tr2']['channel'][:-1]) != ID2:
                raise InputError('IDs of stations in corr_mat_list[%d] do not match IDs of'
                                 'corr_mat_list[0].' % ind)
        if (cm['stats_tr1']['network']+'.'+
            cm['stats_tr1']['station']+'.'+
            cm['stats_tr1']['location']+'.'+
            cm['stats_tr1']['channel'][:-1]) == ID2:
            if (cm['stats_tr2']['network']+'.'+
                cm['stats_tr2']['station']+'.'+
                cm['stats_tr2']['location']+'.'+
                cm['stats_tr2']['channel'][:-1]) != ID1:
                raise InputError('IDs of stations in corr_mat_list[%d] do not match IDs of'
                                 'corr_mat_list[0].' % ind)
                # flip corrmat
                cm = corr_mat_reverse(cm)
        
        cmd.update({cm['stats_tr1']['channel'][-1]+cm['stats_tr2']['channel'][-1]:cm})
    
    # check wheather all Greenstensor components are present
    for rc in required_comp:
        if rc not in list(cmd.keys()):
            print(rc, 'not present')
            #raise InputError('%s component is missing in corr_mat_list' % rc)

    
    return _rotate_corr_dict(cmd,horiz_only=True)


def _rotate_corr_dict(cmd,horiz_only=False):
    """ Rotate correlation matrices in stream from the EE-EN-EZ-NE-NN-NZ-ZE-ZN-ZZ
    system to the RR-RT-RZ-TR-TT-TZ-ZR-ZT-ZZ system. Input matrices are assumed
    to be of same size and simultaneously sampled.
    :type cmd: dictionary
    :param cmd: dictionary of correlation matrix dictionares as produced by
        :class:`~miic.core.macro.recombine_corr_data` where key names are the
        component types listed above
    :type horiz_only: boolean (default False)
    :param horiz_only: If True, the input combinations in the stream are expected to
    be EE-EN-NE-NN and are rotated into the RR-RT-TR-TT system.
    """
    
    # rotation angles
    # phi1 : counter clockwise angle between E and R(towards second station)
    # the leading -1 accounts for the fact that we rotate the coordinates not a vector within them
    phi1 = - np.pi/180*(90-cmd['EE']['stats']['az'])
    # phi2 : counter clockwise angle between E and R(away from first station)
    phi2 = - np.pi/180*(90-cmd['EE']['stats']['baz']+180)
    
    c1 = np.cos(phi1)
    s1 = np.sin(phi1)
    c2 = np.cos(phi2)
    s2 = np.sin(phi2)
    
    RR = deepcopy(cmd['EE'])
    RR['stats_tr1']['channel'] = RR['stats_tr1']['channel'][:-1] + 'R'
    RR['stats_tr2']['channel'] = RR['stats_tr2']['channel'][:-1] + 'R'
    RR['stats']['channel'] = RR['stats_tr1']['channel']+'-'+RR['stats_tr2']['channel']
    RR['corr_data'] = (c1*c2*cmd['EE']['corr_data'] - c1*s2*cmd['EN']['corr_data'] - 
                       s1*c2*cmd['NE']['corr_data'] + s1*s2*cmd['NN']['corr_data'])
    
    RT = deepcopy(cmd['EE'])
    RT['stats_tr1']['channel'] = RT['stats_tr1']['channel'][:-1] + 'R'
    RT['stats_tr2']['channel'] = RT['stats_tr2']['channel'][:-1] + 'T'
    RT['stats']['channel'] = RT['stats_tr1']['channel']+'-'+RT['stats_tr2']['channel']
    RT['corr_data'] = (c1*s2*cmd['EE']['corr_data'] + c1*c2*cmd['EN']['corr_data'] - 
                       s1*s2*cmd['NE']['corr_data'] - s1*c2*cmd['NN']['corr_data'])

    TR = deepcopy(cmd['EE'])
    TR['stats_tr1']['channel'] = TR['stats_tr1']['channel'][:-1] + 'T'
    TR['stats_tr2']['channel'] = TR['stats_tr2']['channel'][:-1] + 'R'
    TR['stats']['channel'] = TR['stats_tr1']['channel']+'-'+TR['stats_tr2']['channel']
    TR['corr_data'] = (s1*c2*cmd['EE']['corr_data'] - s1*s2*cmd['EN']['corr_data'] + 
                       c1*c2*cmd['NE']['corr_data'] - c1*s2*cmd['NN']['corr_data'])
    
    TT = deepcopy(cmd['EE'])
    TT['stats_tr1']['channel'] = TT['stats_tr1']['channel'][:-1] + 'T'
    TT['stats_tr2']['channel'] = TT['stats_tr2']['channel'][:-1] + 'T'
    TT['stats']['channel'] = TT['stats_tr1']['channel']+'-'+TT['stats_tr2']['channel']
    TT['corr_data'] = (s1*s2*cmd['EE']['corr_data'] + s1*c2*cmd['EN']['corr_data'] + 
                       c1*s2*cmd['NE']['corr_data'] + c1*c2*cmd['NN']['corr_data'])
    if horiz_only :
        return RR,RT,TR,TT

    elif not horiz_only :
        RZ = deepcopy(cmd['EE'])
        RZ['stats_tr1']['channel'] = RZ['stats_tr1']['channel'][:-1] + 'R'
        RZ['stats_tr2']['channel'] = RZ['stats_tr2']['channel'][:-1] + 'Z'
        RZ['stats']['channel'] = RZ['stats_tr1']['channel']+'-'+RZ['stats_tr2']['channel']
        RZ['corr_data'] = (c1*cmd['EZ']['corr_data'] - s1*cmd['NZ']['corr_data'])
        
        TZ = deepcopy(cmd['EE'])
        TZ['stats_tr1']['channel'] = TZ['stats_tr1']['channel'][:-1] + 'T'
        TZ['stats_tr2']['channel'] = TZ['stats_tr2']['channel'][:-1] + 'Z'
        TZ['stats']['channel'] = TZ['stats_tr1']['channel']+'-'+TZ['stats_tr2']['channel']
        TZ['corr_data'] = s1*cmd['EZ']['corr_data'] + c1*cmd['NZ']['corr_data']
        
        ZR = deepcopy(cmd['EE'])
        ZR['stats_tr1']['channel'] = ZR['stats_tr1']['channel'][:-1] + 'Z'
        ZR['stats_tr2']['channel'] = ZR['stats_tr2']['channel'][:-1] + 'R'
        ZR['stats']['channel'] = ZR['stats_tr1']['channel']+'-'+ZR['stats_tr2']['channel']
        ZR['corr_data'] = c2*cmd['ZE']['corr_data'] - s2*cmd['ZE']['corr_data']
        
        ZT = deepcopy(cmd['EE'])
        ZT['stats_tr1']['channel'] = ZT['stats_tr1']['channel'][:-1] + 'Z'
        ZT['stats_tr2']['channel'] = ZT['stats_tr2']['channel'][:-1] + 'T'
        ZT['stats']['channel'] = ZT['stats_tr1']['channel']+'-'+ZT['stats_tr2']['channel']
        ZT['corr_data'] = s2*cmd['ZE']['corr_data'] + c2*cmd['ZN']['corr_data']

        return RR,RT,RZ,TR,TT,TZ,ZR,ZT,deepcopy(cmd['ZZ'])


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

    # starttime = convert_time([corr_mat['stats']['starttime']])[0]
    # endtime = convert_time([corr_mat['stats']['endtime']])[0]
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
    # dv['corr_start'] = stats['corr_start']
    dv['stats'] = stats

    return dv


def corr_mat_correct_stretch(corr_mat, dv):
    """Correct stretching of correlation matrix

    In the case of a homogeneous subsurface velocity change the correlation
    traces are stretched or compressed. This stretching can be measured
    with `corr_mat_stretch`. The resulting `dv` dictionary can be passed to
    this function to remove the stretching from the correlation matrix.

    :type corr_mat: dictionary
    :param corr_mat: correlation matrix dictionary as produced by
        :class:`~miic.core.macro.recombine_corr_data`
    :type dv: Dictionary
    :param dv: velocity change dictionary

    :rtype: Dictionary
    :return: corrected correlation matrix dictionary
    """

    # check input
    if corr_mat_check(corr_mat)['is_incomplete']:
        raise ValueError("Error: corr_mat is not a valid correlation_matix \
            dictionary.")

    if dv_check(dv)['is_incomplete']:
        raise ValueError("Error: dv is not a valid Velocity change\
            dictionary.")

    ccorr_mat = deepcopy(corr_mat)
    ccorr_mat['corr_data'] = time_stretch_apply(ccorr_mat['corr_data'],-1.*dv['value'])

    return ccorr_mat


def corr_mat_shift(
    data: np.ndarray, stats: CorrStats, ref_trc: np.ndarray = None,
    tw: List[np.ndarray] = None, shift_range: float = 10,
    shift_steps: int = 100, sides: str = 'both',
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

    dta = stats.start_lag
    dte = stats.end_lag

    # format (trimm) the matrix for zero-time to be either at the beginning
    # or at the center as required by
    # miic.core.stretch_mod.time_stretch_estimate

    # In case a reference is provided but the matrix needs to be trimmed the
    # reference also needs to be trimmed. To do so we append the reference to
    # the matrix, trimm it and remove the reference again
    if ref_trc != None:
        data = np.concatenate((data,
                            np.atleast_2d(ref_trc)), 0)

    # trim the marices
    if sides == "single":
        # extract the time>0 part of the matrix
        corr_mat = corr_mat_trim(data, stats, 0, dte)
    else:
        # extract the central symmetric part (if dt<0 the trim will fail)
        dt = min(dta, dte)
        corr_mat = corr_mat_trim(data, stats, -dt, dt)

    # create or extract references
    if ref_trc == None:
        ref_trc = corr_mat_extract_trace(data, stats)
    else:
        # extract and remove references from corr matrix again
        ref_trc = data[-1, :]
        cdata = data[:-1, :]
    
    # set sides
    if sides == 'both':
        ss = False
    elif sides == 'right':
        ss = True
    else:
        raise ValueError("Error: side is not recognized. Use either both or\
                            right.")

    dt = time_shift_estimate(corr_mat['corr_data'],
                      ref_trc=ref_trc,
                      tw=tw,
                      shift_range=shift_range,
                      shift_steps=shift_steps,
                      single_sided=ss,
                      return_sim_mat=return_sim_mat)

    # add the keys the can directly be transferred from the correlation matrix
    dt['corr_start'] = stats['corr_start']
    dt['stats'] = stats
    return dt


def corr_mat_correct_shift(corr_mat, dt):
    """Correct shift of a correlation matrix

    In the case of a clock error the correlation traces are shifted in lag time
    This shifting can be measured with `corr_mat_shift`. The resulting `dt` 
    dictionary can be passed to this function to remove the shifhting from the
    correlation matrix.

    :type corr_mat: dictionary
    :param corr_mat: correlation matrix dictionary as produced by
        :class:`~miic.core.macro.recombine_corr_data`
    :type dt: Dictionary
    :param dt: velocity change dictionary

    :rtype: Dictionary
    :return: corrected correlation matrix dictionary
    """

    ccorr_mat = deepcopy(corr_mat)
    ccorr_mat['corr_data'] = time_shift_apply(ccorr_mat['corr_data'],-1.*dt['value'])

    return ccorr_mat

# def corr_mat_add_lat_lon_ele(corr_mat, coord_df):
#     """ Add coordinates to correlation matrix.

#     Add the geographical information provided in ``coord_df`` as a
#     :class:`~pandas.DataFrame` and add it to the correlation matrix. It also
#     accepts input of a correlation trace dictionary.

#     :type corr_data: dictionay
#     :param: corr_data: correlation matrix
#     :type coord_df: :class:`~pandas.DataFrame`
#     :param coord_df: dataframe containing seedID as index and columns with
#         latitude, longitude and elevation

#     :rtype: dictionay
#     :return: **corr_mat_geo**: correlation matrix with the added information
#     """

#     # check input
#     if (_check_stats(corr_mat['stats'])):
#         raise ValueError("Error: corr_mat has incomplete stats.")

#     # fill stats_tr1 and stats_tr2
#     stats_names = ['stats_tr1', 'stats_tr2']
#     for stats_name in stats_names:
#         if corr_mat[stats_name]['location'] == []:
#             corr_mat[stats_name]['location'] = ''
#         if corr_mat[stats_name]['channel'] == []:
#             corr_mat[stats_name]['channel'] = ''
#         tr1_id = corr_mat[stats_name]['network'] + '.' + \
#             corr_mat[stats_name]['station'] + '.' + \
#             corr_mat[stats_name]['location'] + '.' + \
#             corr_mat[stats_name]['channel']
#         selector = _Selector(tr1_id)
#         geo_info = coord_df.select(selector, axis=0)

#         corr_mat[stats_name]['stla'] = geo_info['latitude'][0]
#         corr_mat[stats_name]['stlo'] = geo_info['longitude'][0]
#         corr_mat[stats_name]['stel'] = geo_info['elevation'][0]

#     # copy to stats dict
#     corr_mat['stats']['stla'] = corr_mat[stats_names[0]]['stla']
#     corr_mat['stats']['stlo'] = corr_mat[stats_names[0]]['stlo']
#     corr_mat['stats']['stel'] = corr_mat[stats_names[0]]['stel']
#     corr_mat['stats']['evla'] = corr_mat[stats_names[1]]['stla']
#     corr_mat['stats']['evlo'] = corr_mat[stats_names[1]]['stlo']
#     corr_mat['stats']['evel'] = corr_mat[stats_names[1]]['stel']

#     # calculate relative information
#     try:
#         from obspy.geodetics import gps2dist_azimuth
#     except ImportError:
#         print("Missed obspy funciton gps2dist_azimuth")
#         print("Update obspy.")
#         return

#     dist, az, baz = gps2dist_azimuth(corr_mat['stats']['stla'], \
#                                     corr_mat['stats']['stlo'], \
#                                     corr_mat['stats']['evla'], \
#                                     corr_mat['stats']['evlo'])
#     corr_mat['stats']['az'] = az
#     corr_mat['stats']['baz'] = baz
#     corr_mat['stats']['dist'] = dist / 1000  # conversion to km

#     corr_mat_geo = corr_mat

#     return corr_mat_geo
    

# def corr_mat_create_from_traces(base_dir, save_dir, corr_length=0,
#                                 basename='trace', suffix='',
#                                 networks='*', stations='*',
#                                 locations='*', channels='*',
#                                 delete_trace_files=False):
#     """ Create correlation matrix files from a set or correlation trace files

#     Search the directory ``base_dir`` for files matching the following pattern
#     ``time_basename_stations.networks.locations.channels_suffix.mat``
#     where time is a string indicating the time used to construct the
#     respective correlation trace and the other parts refer to the seedID and
#     can be resticted by setting the keyword arguments according to glob. If
#     the default ``*`` is used every network, station, location, channel
#     combination will be used. If files are found that differ only in the
#     ``time`` part of the filename the are grouped together and put in a
#     correlation matrix which is saved in ``save_dir``. The length of the
#     traces in the correlation matrix to be saved can be set to
#     ``corr_length``. This function returns nothing it saves the date in files.

#     :type base_dir: string
#     :param base_dir: Where the corr traces are stored
#     :type save_dir: string
#     :param save_dir: Where all the corr matrices will be saved
#     :type corr_length: float
#     :param corr_length: length of correlation traces to be saves in seconds
#     :type basename: string
#     :param basename: Common "root" for every generated filename. It must not
#         include underscores
#     :type suffix: string
#     :param suffix: Optional suffix for the filename


#     :type networks: string
#     :param networks: identification of the network combination e.g. 'LT-LT'
#     :type stations: string
#     :param stations: identification of the station combination e.g. 'LT01-LT02'
#     :type locations: string
#     :param locations: identification of the location combination e.g. '0-0'
#     :type channels: string
#     :param channels: identification of the channel combination e.g. 'HHZ-HHZ'

#     :type delete_trace_files: Bool
#     :param delete_trace_files: if True all files whos tarces are put in matrices
#         are deleted.

#     """
#     default_var_name = 'corr_trace'
#     out_file_tag = 'mat'

#     print_time_format = "%Y-%m-%d %H:%M:%S.%f"

#     # filename pattern matching the input parameters
#     if suffix != '':
#         fpattern = '*' + '_' + basename + '_' + \
#         stations + '.' + networks + '.' + locations + '.' + channels + \
#         '_' + suffix + '.mat'
#     else:
#         fpattern = '*' + '_' + basename + '_' + \
#         stations + '.' + networks + '.' + locations + '.' + channels + '.mat'

#     print('Searching for %s' % fpattern)
#     # find the file list matching the pattern
#     flist = sorted(glob1(base_dir, fpattern))
#     # split the filenames in the list into a variable part (time) and a
#     # constant part (base_name_ID_suffix)
#     IDstr = []
#     timestr = []
#     for tfname in flist:
#         ttimestr, tIDstr = tfname.split('_', 1)
#         IDstr.append(tIDstr)
#         timestr.append(ttimestr)
#     # find list with unique entries of IDs
#     combinations = sorted(list(set(IDstr)))
#     # construct output file name
#     ofname = []
#     for tcombination in combinations:
#         spl = tcombination.split('_')
#         ID = ''
#         for tmp in spl[1:-1]:
#             ID += tmp
#         ofname.append(out_file_tag + '_' + ID + '_' + spl[-1])

#     # check file name type
#     # old_style: YYYY-MM-DD_basename_STA1-STA1_CHAN1-CHAN2_suffix.mat
#     # new_style:
#     #    YYYYMMDDThhmmssssssZ_basename_networks.stations.locations.channels_suffix.mat

#     if flist == []:  # try old_style filename
#         print('Assume old style filename')
#         # filename pattern matching the input parameters for old style
#         # file names
#         fpattern = '*' + '_' + basename + '_' + stations + '_' + channels + \
#             '_' + suffix + '.mat'
#         # find the file list matching the pattern
#         flist = sorted(glob1(base_dir, fpattern))
#         var_name = []
#         for tfname in flist:
#             ttimestr, tIDstr = flist[0].split('_', 1)
#             IDstr.append(tIDstr)
#             timestr.append(ttimestr)
#             # in the old style the variable was named according to the
#             # combination. To chop the last
#             var_name.append(tIDstr[-1::-1].split('_', 1)[1][-1::-1])
#         # find list with unique entries of IDs
#         combinations = sorted(list(set(IDstr)))
#         for tcombination in combinations:
#             spl = tcombination.split('_')
#             stat = ''
#             for tmp in spl[1:-2]:
#                 stat += tmp
#             ofname.append(out_file_tag + '_.' + stat.replace('-', '') + \
#                           '..' + spl[-2].replace('-', '') + '_' + spl[-1])

#     # Collect the files
#     # loop over all the combinations that match the conditions (different IDs)
#     for comb_idx, tcombination in enumerate(combinations):
#         X = None
#         time_vect = []
#         sampling_rate = None
#         files_used = []
#         # loop over all the files found in base_dir
#         for ii, tIDstr in enumerate(IDstr):

#             # if the file belongs to the current combination
#             if tIDstr == tcombination:

#                 # load the file
#                 cfilename = join(base_dir,timestr[ii] + '_' + tIDstr)
#                 dat = mat_to_ndarray(cfilename)
#                 dat['stats'] = flatten_recarray(dat['stats'])

#                 # check for consistent sampling rates
#                 checkflag = 0
#                 # initialize meta data when treating the first trace
#                 if sampling_rate == None:
#                     sampling_rate = dat['stats']['sampling_rate']
#                     stats = dat['stats']  # this (starttime, npts) needs to be
#                                           # adopted if the trace length
#                                           # changes
#                     stats_tr1 = dat['stats_tr1']
#                     stats_tr2 = dat['stats_tr2']
#                     starttime = dat['stats']['starttime']
#                     npts = dat['stats']['npts']
#                     # trace_length = int(corr_length*sampling_rate)
#                     # ctime = (arange(0,trace_length,1,dtype=np.float)- \
#                     #    (np.float(trace_length)-1)/2.)/sampling_rate
#                     # print sampling_rate, trace_length
#                 # check sampling rates
#                 if sampling_rate != dat['stats']['sampling_rate']:
#                     checkflag += 1
#                     print('Sampling rate of %s does not match. \
#                         Trace not included.' % timestr[ii] + '_' + tIDstr)

#                 # check correlation times
#                 # this need to be changed because the zero lag time should be
#                 # in the at the 1 1 1971
#                 if starttime != dat['stats']['starttime']:
#                     checkflag += 1
#                     print('Time base of %s does not match. \
#                         Trace not included.' % timestr[ii] + '_' + tIDstr)

#                 # print dat['stats']['sampling_rate'], starttime, \
#                     dat['stats']['starttime'], \
#                     dat['stats_tr1']['starttime'], \
#                     npts, dat['stats']['npts']

#                 if npts != dat['stats']['npts']:
#                     checkflag += 1
#                     print('Trace length of %s does not match. \
#                         Trace not included.' % timestr[ii] + '_' + tIDstr)
#                 # more checks possible but we believe it is covered by the
#                 # filename

#                 if checkflag == 0:
#                     # extract the trace variable
#                     # tr_pattern = 'trace_' + fpattern + '_' + \
#                     # self.channels_pair
#                     # if tr_pattern in load_var1:
#                     #    result1 = dict_sel(load_var1, tr_pattern)
#                     if default_var_name in dat:
#                         result1 = select_var_from_dict(dat, default_var_name)
#                     else:
#                         try:
#                             result1 = select_var_from_dict(dat, var_name[ii])
#                         except:
#                             print('Variable not found')
#                             pass

#                     # keep only central part of length if given
#                     if corr_length != 0:
#                         print('stats time', stats['sampling_rate'])
#                         trace_length = int(corr_length * sampling_rate)
#                         print('tra len', trace_length)
#                         result1 = nd_mat_center_part(result1, trace_length,
#                                                      axis=0)
#                         timeOffset = timedelta(np.ceil(
#                                       np.float(npts - trace_length) / 2.) / \
#                                                sampling_rate / 86400)
#                         print('tO', timeOffset)
#                         stats['starttime'] = convert_time_to_string([timeOffset
#                                 + convert_time([stats['starttime']])[0]])[0]
#                         stats['npts'] = trace_length
#                         print('stats time2', stats['starttime'])
#                     # estimate time variable
#                     t1 = dat['stats_tr1']['starttime']
#                     time_tr1 = convert_time([t1])[0]

#                     t2 = dat['stats_tr2']['starttime']
#                     time_tr2 = convert_time([t2])[0]

#                     mtime = '%s' % (max(time_tr1,
#                                         time_tr2).strftime(print_time_format))

#                     time_vect.append(mtime)

#                     if not isinstance(X,np.ndarray):
#                         X = np.atleast_2d(result1).T
#                     else:
#                         X = np.vstack((X, np.atleast_2d(result1).T))

#                     files_used.append(cfilename)

#         corr_mat = {'corr_data': X,
#                     'time': time_vect,
#                     'stats': stats,
#                     'stats_tr1': stats_tr1,
#                     'stats_tr2': stats_tr2}

#         # keep only central part of the correlation matrix
#         # (this is better done during stacking above)
#         # if not corr_length == 0:
#         #    corr_mat = corr_mat_trim(corr_mat,-corr_length,corr_length)

#         # save the matlab variable

#         #print 'saving file %s of size %d x %d' % \
#         #    (ofname[comb_idx], corr_mat['corr_data'].shape[0],
#         #     corr_mat['corr_data'].shape[1])

#         savemat(join(save_dir, ofname[comb_idx]), corr_mat, oned_as='row')

#         # delete files
#         if delete_trace_files:
#             for del_file in files_used:
#                 os.remove(del_file)
