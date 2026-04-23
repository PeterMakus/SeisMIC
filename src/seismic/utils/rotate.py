'''
:copyright:
   The SeisMIC development team (jlehr@gfz.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Johanna Lehr (jlehr@gfz.de)

Created: Thursday, 16th April 2026 06:49:10 pm
Last Modified: Thursday, 23rd April 2026 06:01:22 pm
'''

import numpy as np
from obspy import Stream


def rotate_corrstream_enz2rtz(st: Stream) -> Stream:
    """ Rotate traces in stream from the EE-EN-EZ-NE-NN-NZ-ZE-ZN-ZZ system to
    the RR-RT-RZ-TR-TT-TZ-ZR-ZT-ZZ system using the explicit expressions
    derived in derive_rotation_expressions.ipynb. The input stream must have
    9 traces with channel combinations EE, EN, EZ, NE, NN, NZ, ZE, ZN, ZZ.

    Parameters
    ----------
    st: obspy.Stream
        Stream with 9 traces of pairwise correlations in the ENZ system.
        Sorting is applied to this input. Az and baz must be set in the
        trace stats.

    Returns
    -------
    obspy.Stream
        New stream with 9 traces of pairwise correlations in the RTZ system. The
        order of the traces is RR, RT, RZ, TR, TT, TZ, ZR, ZT, ZZ.

    Note
    -----
    - Sorting is applied to the input stream to ensure that the components are
    in the expected order.
    - Returns a copy of the input stream with rotated traces.

    Technical remarks
    --------------------
    It is faster to compute every rotated trace separately using the explicit
    expressions than to compute the rotation matrix and apply it to the data.
    Converting the stream to a 9xN numpy array and back is more time
    consuming than computing the rotated traces separately.
    """
    # Check that the stream contains 9 traces
    if len(st) != 9:
        raise ValueError(
            "Expected 9 traces in the input stream, but got %d" % len(st))

    st.sort(keys=['channel'])

    # Check that the stream contains the expected channels
    expected_channels = ['EE', 'EN', 'EZ', 'NE', 'NN', 'NZ', 'ZE', 'ZN', 'ZZ']
    for i, chan in enumerate(expected_channels):
        if (st[i].stats.channel[2] != chan[0] or
                st[i].stats.channel[6] != chan[1]):
            raise ValueError(
                "Expected channel %s at position %d, but got %s" % (
                    chan, i, st[i].stats.channel))

    # rotation angles
    # phi1: (positive) angle between N and R at first station
    # (away from first station)
    phi1 = np.radians(st[0].stats.az)
    # phi2: (positive) angle between N and R at second station
    # (away from first station)
    phi2 = np.radians(st[0].stats.baz + 180)

    c1 = np.cos(phi1)
    s1 = np.sin(phi1)
    c2 = np.cos(phi2)
    s2 = np.sin(phi2)

    rtz = Stream()
    RR = st[0].copy()
    RR.data = s1*s2*st[0].data + s1*c2*st[1].data + c1*s2*st[3].data +\
        c1*c2*st[4].data
    tcha = list(RR.stats['channel'])
    tcha[2] = 'R'
    tcha[6] = 'R'
    RR.stats['channel'] = ''.join(tcha)
    rtz.append(RR)

    RT = st[0].copy()
    RT.data = s1*c2*st[0].data - s1*s2*st[1].data + c1*c2*st[3].data -\
        c1*s2*st[4].data
    tcha = list(RT.stats['channel'])
    tcha[2] = 'R'
    tcha[6] = 'T'
    RT.stats['channel'] = ''.join(tcha)
    rtz.append(RT)

    RZ = st[0].copy()
    RZ.data = s1*st[2].data + c1*st[5].data
    tcha = list(RZ.stats['channel'])
    tcha[2] = 'R'
    tcha[6] = 'Z'
    RZ.stats['channel'] = ''.join(tcha)
    rtz.append(RZ)

    TR = st[0].copy()
    TR.data = c1*s2*st[0].data + c1*c2*st[1].data - s1*s2*st[3].data -\
        s1*c2*st[4].data
    tcha = list(TR.stats['channel'])
    tcha[2] = 'T'
    tcha[6] = 'R'
    TR.stats['channel'] = ''.join(tcha)
    rtz.append(TR)

    TT = st[0].copy()
    TT.data = c1*c2*st[0].data - c1*s2*st[1].data - s1*c2*st[3].data +\
        s1*s2*st[4].data
    tcha = list(TT.stats['channel'])
    tcha[2] = 'T'
    tcha[6] = 'T'
    TT.stats['channel'] = ''.join(tcha)
    rtz.append(TT)

    TZ = st[0].copy()
    TZ.data = c1*st[2].data - s1*st[5].data
    tcha = list(TZ.stats['channel'])
    tcha[2] = 'T'
    tcha[6] = 'Z'
    TZ.stats['channel'] = ''.join(tcha)
    rtz.append(TZ)

    ZR = st[0].copy()
    ZR.data = s2*st[6].data + c2*st[7].data
    tcha = list(ZR.stats['channel'])
    tcha[2] = 'Z'
    tcha[6] = 'R'
    ZR.stats['channel'] = ''.join(tcha)
    rtz.append(ZR)

    ZT = st[0].copy()
    ZT.data = c2*st[6].data - s2*st[7].data
    tcha = list(ZT.stats['channel'])
    tcha[2] = 'Z'
    tcha[6] = 'T'
    ZT.stats['channel'] = ''.join(tcha)
    rtz.append(ZT)

    ZZ = st[8].copy()
    rtz.append(ZZ)

    return rtz


def rotate_corrstream_en2rt(st: Stream) -> Stream:
    """ Rotate traces in stream from the EE-EN-NE-NN system to
    the RR-RT-TR-TT system using the explicit expressions
    derived in derive_rotation_expressions.ipynb. The input stream is
    expected to be sorted alphabetically by channel pair, i.e. st[0:4]
    corresponds to EE, EN, NE, NN.

    Parameters
    ----------
    st: obspy.Stream
        Stream with 4 traces of pairwise correlations in the EN system. Sorting
        is applied to this input. Az and baz must be set in the trace stats.

    Returns
    -------
    obspy.Stream
        New stream with 4 traces of pairwise correlations in the RT system. The
        order of the traces is RR, RT, TR, TT.

    Remarks
    -------
    It is faster to compute every rotated trace separately using the explicit
    expressions than to compute the rotation matrix and apply it to the data.
    Converting the stream to a 4xN numpy array and back is more time
    consuming than computing the rotated traces separately.
    """
    # Check that the stream contains 4 traces
    if len(st) != 4:
        raise ValueError(
            "Expected 4 traces in the input stream, but got %d" % len(st))

    st.sort(keys=['channel'])

    # Check that the stream contains the expected channels
    expected_channels = ['EE', 'EN', 'NE', 'NN']
    for i, chan in enumerate(expected_channels):
        if (st[i].stats.channel[2] != chan[0] or
                st[i].stats.channel[6] != chan[1]):
            raise ValueError(
                "Expected channel %s at position %d, but got %s" % (
                    chan, i, st[i].stats.channel))

    # rotation angles
    # phi1: (positive) angle between N and R at first station
    # (away from first station)
    phi1 = np.radians(st[0].stats.az)
    # phi2: (positive) angle between N and R at second station
    # (away from first station)
    phi2 = np.radians(st[0].stats.baz + 180)

    c1 = np.cos(phi1)
    s1 = np.sin(phi1)
    c2 = np.cos(phi2)
    s2 = np.sin(phi2)

    rt = Stream()
    RR = st[0].copy()
    RR.data = s1*s2*st[0].data + s1*c2*st[1].data + c1*s2*st[2].data +\
        c1*c2*st[3].data
    tcha = list(RR.stats['channel'])
    tcha[2] = 'R'
    tcha[6] = 'R'
    RR.stats['channel'] = ''.join(tcha)
    rt.append(RR)

    RT = st[0].copy()
    RT.data = s1*c2*st[0].data - s1*s2*st[1].data + c1*c2*st[2].data -\
        c1*s2*st[3].data
    tcha = list(RT.stats['channel'])
    tcha[2] = 'R'
    tcha[6] = 'T'
    RT.stats['channel'] = ''.join(tcha)
    rt.append(RT)

    TR = st[0].copy()
    TR.data = c1*s2*st[0].data + c1*c2*st[1].data - s1*s2*st[2].data -\
        s1*c2*st[3].data
    tcha = list(TR.stats['channel'])
    tcha[2] = 'T'
    tcha[6] = 'R'
    TR.stats['channel'] = ''.join(tcha)
    rt.append(TR)

    TT = st[0].copy()
    TT.data = c1*c2*st[0].data - c1*s2*st[1].data - s1*c2*st[2].data +\
        s1*s2*st[3].data
    tcha = list(TT.stats['channel'])
    tcha[2] = 'T'
    tcha[6] = 'T'
    TT.stats['channel'] = ''.join(tcha)
    rt.append(TT)

    return rt


def rotate_multi_corr_stream(st: Stream) -> Stream:
    """Rotate a stream with full Greens tensor from ENZ to RTZ

    Take a stream with numerous correlation traces and rotate the
    combinations of ENZ components into combinations of RTZ components in
    all nine components of the Green's tensor are present. If not all nine
    components are present no trace for this station combination is returned.

    :type st: obspy.stream
    :param st: stream with data in ENZ system
    :rtype: obspy.stream
    :return: stream in the RTZ system
    """

    raise NotImplementedError(
        "Rotation of multi-correlation stream is not implemented yet.")
#     out_st = Stream()
#     while st:
#         tl = list(range(9))
#         tst = st.select(network=st[0].stats['network'],
#                         station=st[0].stats['station'])
#         cnt = 0
#         for ttr in tst:
#             if ttr.stats['channel'][2] == 'E':
#                 if ttr.stats['channel'][6] == 'E':
#                     tl[0] = ttr
#                     cnt += 1
#                 elif ttr.stats['channel'][6] == 'N':
#                     tl[1] = ttr
#                     cnt += 2
#                 elif ttr.stats['channel'][6] == 'Z':
#                     tl[2] = ttr
#                     cnt += 4
#             elif ttr.stats['channel'][2] == 'N':
#                 if ttr.stats['channel'][6] == 'E':
#                     tl[3] = ttr
#                     cnt += 8
#                 elif ttr.stats['channel'][6] == 'N':
#                     tl[4] = ttr
#                     cnt += 16
#                 elif ttr.stats['channel'][6] == 'Z':
#                     tl[5] = ttr
#                     cnt += 32
#             elif ttr.stats['channel'][2] == 'Z':
#                 if ttr.stats['channel'][6] == 'E':
#                     tl[6] = ttr
#                     cnt += 64
#                 elif ttr.stats['channel'][6] == 'N':
#                     tl[7] = ttr
#                     cnt += 128
#                 elif ttr.stats['channel'][6] == 'Z':
#                     tl[8] = ttr
#                     cnt += 256
#         if cnt == 2**9-1:
#             st0 = Stream()
#             for t in tl:
#                 st0.append(t)
#             st1 = _rotate_corr_stream(st0)
#             out_st += st1
#         elif cnt == 27:  # only horizontal component combinations present
#             st0 = Stream()
#             for t in [0, 1, 3, 4]:
#                 st0.append(tl[t])
#             st1 = _rotate_corr_stream_horizontal(st0)
#             out_st += st1
#         elif cnt == 283:  # horizontal combinations + ZZ
#             st0 = Stream()
#             for t in [0, 1, 3, 4]:
#                 st0.append(tl[t])
#             st1 = _rotate_corr_stream_horizontal(st0)
#             out_st += st1
#             out_st.append(tl[8])
#         for ttr in tst:
#             for ind, tr in enumerate(st):
#                 if ttr.id == tr.id:
#                     st.pop(ind)

#     return out_st
