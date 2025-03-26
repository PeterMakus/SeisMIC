'''
Module containing functions for preprocessing in the frequency domain


:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    `EUROPEAN UNION PUBLIC LICENCE v. 1.2
    <https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>`_
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 03:40:11 pm
Last Modified: 2025-03-13 14:11:41 (J. Lehr)
'''
from copy import deepcopy
import logging

import numpy as np
import obspy.signal as osignal

from ..utils import processing_helpers as ph
from .. import logfactory

parentlogger = logfactory.create_logger()
module_logger = logging.getLogger(parentlogger.name+".preprocessing_fd")


functions_acception_jointnorm = ['spectralWhitening']


def FDfilter(B: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    Filter Fourier-transformed data

    Filter Fourier tranformed data by tapering in frequency domain. The `args`
    dictionary is supposed to contain the key `flimit` with a value that is a
    four element list or tuple defines the corner frequencies (f1, f2, f3, f4)
    in Hz of the cosine taper which is one between f2 and f3 and tapers to zero
    for f1 < f < f2 and f3 < f < f4.

    :type B: numpy.ndarray
    :param B: Fourier transformed time series data with frequency oriented\\
        along the first dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: params['freqs'] contains an array with the freqency values
        of the samples in `B`

    :rtype: numpy.ndarray
    :return: filtered spectal data
    """
    args = deepcopy(args)
    args.update({'freqs': params['freqs']})
    tap = osignal.invsim.cosine_taper(B.shape[1], **args)
    B *= np.tile(np.atleast_2d(tap), (B.shape[0], 1))
    return B


def FDsignBitNormalization(
        B: np.ndarray, args: dict, params: dict) -> np.ndarray:
    """
    Sign bit normalization of frequency transformed data

    Divides each sample by its amplitude resulting in trace with amplidues of
    (-1, 0, 1). As this operation is done in frequency domain it requires two
    Fourier transforms and is thus quite costly but alows to be performed
    after other steps of frequency domain procesing e.g. whitening.


    :type B: numpy.ndarray
    :param B: Fourier transformed time series data with frequency oriented\\
        along the first dimension (columns)
    :type args: dictionary
    :param args: not used in this function
    :type params: dictionary
    :param params: not used in this function

    :rtype: numpy.ndarray
    :return: frequency transform of the 1-bit normalized data
    """
    B = np.fft.irfft(B)
    # np.sign only takes the real part into account
    C = np.sign(B)
    return np.fft.rfft(C)


def spectralWhitening(B: np.ndarray, args: dict, params) -> np.ndarray:
    """
    Spectal whitening of Fourier-transformed date

    Normalize the amplitude spectrum of the complex spectra in `B` by its
    absolute mean. The
    `args` dictionary may contain the keyword `joint_norm`. Its values can be
    one of 1, 2, or 3 or True/False.
    A value of 1 is equivalent to False and means no normalization is applied.
    A value of 2 or 3 means that 2 or 3 consecutive rows of the matrix are
    normalized jointly by the mean of their amplitude spectra. If set to
    `True`, we assume 3. This is useful for later rotation of
    correlated traces in the ZNE system into the ZRT system. Normalization
    over 2 is useful if only NE components are available.

    :type B: numpy.ndarray
    :param B: Fourier transformed time series data with frequency oriented\\
        along the first dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: not used here

    :rtype: numpy.ndarray
    :return: whitened spectal data

    Warning
    -------
    The function normalizes pairs/triplets of rows. It has no way of knowing
    if that is a sensible thing to do. It is up to the user to ensure that the
    components of one station are next to each other and that there are always
    pairs/triplets present.
    """
    module_logger.debug("Spectral whitening: shape of matrix: %s", B.shape)
    absB = np.absolute(B)
    ph.get_joint_norm(absB, args)
    with np.errstate(invalid='raise'):
        try:
            B = np.true_divide(B, absB)
        except FloatingPointError as e:
            errargs = np.argwhere(absB == 0)
            # Report error where there is zero divides for a non-zero freq
            if not np.all(errargs[:, 0] == 0):
                logging.debug(f'{e} {errargs}')
    # Set zero frequency component to zero
    B[:, 0] = 0.j

    return B
