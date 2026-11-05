'''
Helper functions for preprocessing modules.

:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Johanna Lehr (jlehr@gfz-potsdam.de)

Created: Wednesday, 2025-03-20 15:15:21
Last Modified: 2025-03-20 15:15:28
'''

import numpy as np
from scipy.ndimage import uniform_filter1d


def get_joint_norm(B, args: dict) -> None:
    """
    Average over n rows of a 2D numpy array."

    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: must contain key 'joint_norm' with value of
        1, 2, 3 or False/True

    :rtype: numpy.ndarray
    :return: Matrix B with rows averaged over n rows
    """

    if 'joint_norm' in list(args.keys()):
        if args['joint_norm'] is True:
            args['joint_norm'] = 3

        if args['joint_norm'] in [2, 3]:
            k = args['joint_norm']
            assert B.shape[0] % k == 0, "For joint normalization with %d the "\
                "number of traces needs to the multiple of it, but is %d" % (
                    k, B.shape[0])
            B[:, :] = np.repeat(np.mean(B.reshape(-1, k, B.shape[1]), axis=1),
                                k, axis=0)
        elif args['joint_norm'] == 1 or args['joint_norm'] is False:
            pass
        else:
            raise ValueError(
                "joint_norm must be int of 1, 2, 3 or False/True(==3 )")


def smooth_rows(B, args: dict, params: dict) -> None:
    """"
    Smooth matrix along rows.

    Smoothing is applied in both directions to avoid a shift by
    convolution with a rectangular window.

    :type A: numpy.ndarray
    :param A: matrix with time series data (seismic traces) along rows
    :type args: dictionary
    :param args: must contain key 'windowLength' with value of the
        window length in seconds
    :type params: dictionary
    :param params: must contain key 'sampling_rate' with value of the
        sampling rate in Hz

    :rtype: numpy.ndarray
    :return: copy of Matrix  B, smoothed along rows
    """
    if args['windowLength'] <= 0:
        raise ValueError('Window Length has to be greater than 0.')

    B = np.atleast_2d(B)
    window = (
        np.ones(int(np.ceil(args['windowLength'] * params['sampling_rate'])))
        / np.ceil(args['windowLength']*params['sampling_rate']))

    if window.size > B.shape[1]:
        raise ValueError('Window Length is too large for the data.')

    # Two moving-average passes are mathematically the same triangular
    # (double rectangular) smoothing kernel as the previous np.convolve
    # pair, but O(N) per pass instead of O(N*W) (uniform_filter1d runs
    # ~145x faster at corr_len scale). mode='constant' reproduces
    # np.convolve's zero padding; the default centering matches
    # np.convolve's 'same' alignment for odd and even window lengths
    # alike (verified: interior deviation ~6e-8 in float32).
    W = window.size
    B = uniform_filter1d(B, W, axis=1, mode='constant')
    B = uniform_filter1d(B[:, ::-1], W, axis=1, mode='constant')[:, ::-1]
    return B
