'''
Module containing functions for preprocessing in the frequency domain


:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3 <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 03:40:11 pm
Last Modified: Tuesday, 20th July 2021 03:43:11 pm
'''
from copy import deepcopy


import numpy as np
import obspy.signal as osignal


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
    tap = osignal.invsim.cosine_taper(B.shape[0], **args)
    B *= np.tile(np.atleast_2d(tap).T, (1, B.shape[1]))
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
    B = np.fft.irfft(B, axis=0)
    # B should always be real, so this line does not make an awful lot of sense
    C = np.sign(B.real)
    return np.fft.rfft(C, axis=0)
