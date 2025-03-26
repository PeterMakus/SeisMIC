'''
Module that contains functions for preprocessing in the time domain

:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    `EUROPEAN UNION PUBLIC LICENCE v. 1.2
    <https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>`_
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 03:24:01 pm
Last Modified: 2025-03-13 14:12:06 (J. Lehr)
'''
from copy import deepcopy

import numpy as np
from scipy.fftpack import next_fast_len
from scipy import signal
from scipy.signal import detrend as sp_detrend
import obspy.signal as osignal

from seismic.utils.fetch_func_from_str import func_from_str
from ..utils import processing_helpers as ph

import logging
from .. import logfactory

parentlogger = logfactory.create_logger()
module_logger = logging.getLogger(parentlogger.name+".preprocessing_td")


def clip(A: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    Clip time series data at a multiple of the standard deviation

    Set amplitudes exeeding a certain threshold to this threshold.
    The threshold for clipping is estimated as the standard deviation of each
    trace times a factor specified in `args`.

    :Note: Traces should be demeaned before clipping.

    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: the only keyword allowed is `std_factor` describing the \\
        scaling of the standard deviation for the clipping threshold
    :type params: dictionary
    :param params: not used here

    :rtype: numpy.ndarray
    :return: clipped time series data
    """
    stds = np.nanstd(A, axis=1)
    for ind in range(A.shape[0]):
        ts = args['std_factor']*stds[ind]
        A[ind, A[ind, :] > ts] = ts
        A[ind, A[ind, :] < -ts] = -ts
    return A


def detrend(A: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    Detrend wrapper. Can call scipy.signal.detrend or a faster custom function.

    :param A: 1 or 2D array
    :type A: np.ndarray
    :param args: arguments for detrend function
    :type args: dict
    :param params: params file
    :type params: dict
    :raises ValueError: For unknown detrend method
    :raises ValueError: For unknown detrend type
    :return: detrended data
    :rtype: np.ndarray
    """
    method = args.get('method', 'qr')
    if method == 'scipy':
        return detrend_scipy(A, args, params)
    elif method == 'qr':
        detrend_type = args.get('type', 'linear')
        if detrend_type == 'linear':
            return detrendqr(A)
        elif detrend_type == 'constant':
            return demean(A)
        else:
            raise ValueError('Unknown detrend type')
    else:
        raise ValueError('Unknown detrend method')


def detrend_scipy(A: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    Remove trend from data

    .. seealso:: :func:`scipy.signal.detrend` for input arguments
    """
    if np.all(np.isnan(A)):
        return A
    for ii in range(A.shape[0]):
        try:
            A[ii, ~np.isnan(A[ii])] = sp_detrend(
                A[ii, ~np.isnan(A[ii])], axis=-1, overwrite_data=True,
                **args)
        except ZeroDivisionError:
            # When the line is nothing but nans
            pass
    return A


def detrendqr(data: np.ndarray) -> np.ndarray:
    """
    Remove trend from data using QR decomposition. Faster than
    scipy.signal.detrend. Shamelessly adapted from
    NoisePy

    .. seealso:: Adapted from `NoisePy <https://doi.org/10.1785/0220190364>`_

    :param data: 1 or 2D array
    :type data: np.ndarray
    :raises ValueError: For data with more than 2 dimensions
    :return: detrended data
    :rtype: np.ndarray
    """
    npts = data.shape[-1]
    X = np.ones((npts, 2))
    X[:, 0] = np.arange(0, npts, dtype=np.float32) / npts
    Q, R = np.linalg.qr(X)
    rq = np.dot(np.linalg.inv(R), Q.transpose())
    if data.ndim == 1:
        coeff = np.dot(rq, data)
        data = data - np.dot(X, coeff)
    elif data.ndim == 2:
        for ii in range(data.shape[0]):
            coeff = np.dot(rq, data[ii])
            data[ii] = data[ii] - np.dot(X, coeff)
    else:
        raise ValueError('data must be 1D or 2D')
    return data


def demean(data: np.ndarray, args=None, params=None) -> np.ndarray:
    """
    Demean data.

    :param data: 1 or 2D array
    :type data: np.ndarray
    :return: demeaned data
    :rtype: np.ndarray
    """
    return data - np.nanmean(data, axis=-1, keepdims=True)


def mute(A: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    Mute parts of data that exceed a threshold

    To completely surpress the effect of data with high amplitudes e.g. after
    aftershocks these parts of the data are muted (set to zero). The respective
    parts of the signal are identified as those where the envelope in a given
    frequency exceeds a threshold given directly as absolute numer or as a
    multiple of the data's standard deviation. A taper of length `taper_len` is
    applied to smooth the edges of muted segments. Setting `extend_gaps` to
    True will ensure that the taper is applied outside the segments and data
    inside these segments will all zero. Edges of the data will be tapered too
    in this case.



    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first dimension
        (columns)
    :type args: dictionary
    :param args: the following keywords are allowed:

        * `filter`:
            (dictionary) description of filter to be applied before
            calculation of the signal envelope. If not given the envelope is
            calculated from raw data. The value of the keyword filter is the
            same as the `args` for the function `TDfilter`.
        * `threshold`:
            (float) absolute amplitude of threshold for muting
        * `std_factor`:
            (float) alternativly to an absolute number the threhold
            can be estimated as a multiple of the standard deviation if the
            scaling is given in as value of the keyword `std_factor`. If
            neither `threshold` nor `std_factor` are given `std_factor=1` is
            assumed.
        * `extend_gaps`:
            (boolean) if True date above the threshold is
            guaranteed to be muted, otherwise tapering will leak into these
            parts. This step involves an additional convolution.
        * `taper_len`:
            (float) length of taper for muted segments in seconds

    :type params: dictionary
    :param params: filled automatically by `pxcorr`

    :rtype: numpy.ndarray
    :return: clipped time series data

    :Example:
        ``args={'filter':{'type':'bandpass', 'freqmin':1., 'freqmax':6.},
        'taper_len':1., 'threshold':1000, 'std_factor':1, 'extend_gaps':True}``
    """

    if args['taper_len'] == 0:
        raise ValueError('Taper Length cannot be zero.')

    # return zeros if length of traces is shorter than taper
    ntap = int(args['taper_len']*params['sampling_rate'])
    if A.shape[1] <= ntap:
        return np.zeros_like(A)

    # filter if asked to
    if 'filter' in list(args.keys()):
        C = TDfilter(A, args['filter'], params)
    else:
        C = A

    # calculate envelope
    D = np.abs(C)

    # calculate threshold
    if 'threshold' in list(args.keys()):
        thres = np.zeros(A.shape[0]) + args['threshold']
    elif 'std_factor' in list(args.keys()):
        thres = np.std(C, axis=1) * args['std_factor']
    else:
        thres = np.std(C, axis=1)

    # calculate mask
    mask = np.ones_like(D)
    mask[D > np.tile(thres, (D.shape[1], 1)).T] = 0
    # extend the muted segments to make sure the whole segment is zero after
    if args['extend_gaps']:
        tap = np.ones(ntap)/ntap
        for ind in range(A.shape[0]):
            mask[ind, :] = np.convolve(mask[ind, :], tap, mode='same')
        nmask = np.ones_like(D)
        nmask[mask < 1.] = 0
    else:
        nmask = mask

    # apply taper
    tap = 2. - (np.cos(np.arange(ntap, dtype=float)/ntap*2.*np.pi) + 1.)
    tap /= ntap
    for ind in range(A.shape[0]):
        nmask[ind, :] = np.convolve(nmask[ind, :], tap, mode='same')

    # mute data with tapered mask
    A *= nmask
    return A


def normalizeStandardDeviation(
        A: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    Divide the time series by their standard deviation

    Divide the amplitudes of each trace by its standard deviation.
    If `joint_norm` in `args` is set to 2 or 3, the arithmetic mean of the
    standard deviation is computed over 2 or 3 consecutive rows in the array,
    respectively. If set to True, we assume 3 traces. This is useful for later
    rotation of correlated traces in the ZNE system into the ZRT system.
    Normalization over 2 is useful if only NE components are available.
    If not given or set to 1 or False no joint normalization is applied.

    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: `joint_norm` can be 1, 2, 3 or True/False
    :type params: dictionary
    :param params: not used here

    :rtype: numpy.ndarray
    :return: normalized time series data
    """
    std = np.std(A, axis=-1)
    ph.get_joint_norm(std[:, None], args)
    # avoid creating nans or Zerodivisionerror
    std[np.where(std == 0)] = 1
    A = (A.T/std).T
    return A


def signBitNormalization(
        A: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    One bit normalization of time series data

    Return the sign of the samples (-1, 0, 1).

    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: not used here
    :type params: dictionary
    :param params: not used here

    :rtype: numpy.ndarray
    :return: 1-bit normalized time series data
    """
    return np.sign(A)


def taper(A: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    Taper to the time series data

    Apply a simple taper to the time series data.

    `args` has the following structure:

        args = {'type':`type of taper`,taper_args}``

        `type` may be `cosine_taper` with the corresponding taper_args `p` the
        percentage of the traces to taper. Possibilities of `type` are \\
        given by `obspy.signal`.

        :Example:
            ``args = {'type':'cosine_taper','p':0.1}``

    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: not used here

    :rtype: numpy.ndarray
    :return: tapered time series data
    """
    if args['type'] == 'cosine_taper':
        func = osignal.invsim.cosine_taper
    else:
        func = getattr(signal.windows, args['type'])
    args = deepcopy(args)
    args.pop('type')
    tap = func(A.shape[1], **args)
    A *= np.tile(np.atleast_2d(tap), (A.shape[0], 1))
    return A


def TDnormalization(A: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    Amplitude dependent time domain normalization

    Calculate the envelope of the filtered trace, smooth it in a window of
    length `windowLength` and normalize the waveform by this trace. Mandatory
    keywords in `args` are `filter and `windowLength` that describe the
    filter and the length of the envelope smoothing window, respectively.
    If no filtering is desired, `filter` can be an empty dictionary.
    Additionally, `joint_norm` can be set to integers 1, 2, or 3 or True/False.
    If 1 or False no normalization is applied. If 2 or 3, the arithmetic mean
    of the envelope is computed over 2 or 3 consecutive rows in the array,
    respectively. If True, we assume 3 traces. This is useful for later
    rotation of correlated traces in the ZNE system into the ZRT system.
    Normalization over 2 is useful if only NE components are available.


    `args` has the following structure:

        args = {'windowLength':`length of the envelope smoothing window in \\
        [s]`,'filter':{'type':`filterType`, fargs}, 'joint_norm':`joint_norm`}

        `type` may be `bandpass` with the corresponding fargs `freqmin` and \\
        `freqmax` or `highpass`/`lowpass` with the `fargs` `freqmin`/`freqmax`
        `joint_norm` can be 1, 2, 3 or True/False

        :Example:
            ``args = {'windowLength':5,'filter':{'type':'bandpass',
            'freqmin':0.5, 'freqmax':2.}}``

            ``args = {'windowLength':5,'filter':{'type':'bandpass',
            'freqmin':0.5, 'freqmax':2.}, 'joint_norm':3}``

            ``args = {'windowLength':5,'filter':{}, 'joint_norm':True}``

    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: not used here

    :rtype: numpy.ndarray
    :return: normalized time series data
    """

    if args['windowLength'] <= 0:
        raise ValueError('Window Length has to be greater than 0.')
    # filter if args['filter']
    if args['filter']:
        B = TDfilter(A, args['filter'], params)
        # func = getattr(osignal, args['filter']['type'])
        # fargs = deepcopy(args['filter'])
        # fargs.pop('type')
        # B = func(A.T, df=params['sampling_rate'], **fargs).T
    else:
        B = deepcopy(A)
    # simple calculation of envelope
    B = B**2

    # jointly normalize envelope if requested
    ph.get_joint_norm(B, args)

    # smooth envelope (returns 2d array)
    B = ph.smooth_rows(B, args, params)

    # damp the envelope
    B += np.max(B, axis=1)[:, None]*1e-6

    B = np.squeeze(B)

    # normalization
    A /= np.sqrt(B)
    return A


def TDfilter(A: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    Filter time series data

    Filter in time domain. Types of filters are defined by `obspy.signal`.

    `args` has the following structure:

        args = {'type':`filterType`, fargs}

        `type` may be `bandpass` with the corresponding fargs `freqmin` and
        `freqmax` or `highpass`/`lowpass` with the `fargs` `freqmin`/`freqmax`

        :Example:
            ``args = {'type':'bandpass','freqmin':0.5,'freqmax':2.}``

    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: not used here

    :rtype: numpy.ndarray
    :return: filtered time series data
    """
    func = func_from_str('obspy.signal.filter.%s' % args['type'])
    args = deepcopy(args)
    args.pop('type')
    A = func(A, df=params['sampling_rate'], **args)
    return A


def zeroPadding(A: np.ndarray, args: dict, params: dict, axis=1) -> np.ndarray:
    """
    Append zeros to the traces

    Pad traces with zeros to increase the speed of the Fourier transforms and
    to avoid wrap around effects. Three possibilities for the length of the
    padding can be set in ``args['type']``:

    - ``nextFastLen``:
        Traces are padded to a length that is the next fast
        fft length
    - ``avoidWrapAround``:
        depending on length of the trace that is to be used
        the padded part is just long enough to avoid wrap around
    - ``avoidWrapFastLen``:
        Use the next fast length that avoids wrap around

    :Example: ``args = {'type':'avoidWrapPowerTwo'}``

    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: not used here
    :param axis: axis to pad on
    :type axis: tuple, optional

    :rtype: numpy.ndarray
    :return: zero padded time series data
    """
    if A.ndim > 2 or axis > 1:
        raise NotImplementedError('Only two-dimensional arrays are supported.')
    if A.ndim == 1 and len(A) == 0:
        raise ValueError('Input Array is empty')
    npts = A.shape[axis]
    if A.ndim == 2:
        ntrc = A.shape[axis-1]
    elif A.ndim == 1:
        ntrc = 1
    if args['type'] == 'nextFastLen':
        N = next_fast_len(npts)
    elif args['type'] == 'avoidWrapAround':
        N = npts + params['sampling_rate'] * params['lengthToSave']
    elif args['type'] == 'avoidWrapFastLen':
        N = next_fast_len(int(
            npts + params['sampling_rate'] * params['lengthToSave']))
    else:
        raise ValueError("type '%s' of zero padding not implemented" %
                         args['type'])

    if axis == 0:
        A = np.concatenate(
            (A, np.zeros((N-npts, ntrc), dtype=np.float32)), axis=axis)
    else:
        A = np.concatenate(
            (A, np.zeros((ntrc, N-npts), dtype=np.float32)), axis=axis)
    return A


functions_accepting_jointnorm = [
    normalizeStandardDeviation, TDnormalization]
