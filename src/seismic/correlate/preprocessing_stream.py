'''
Module that contains functions for preprocessing on obspy streams

:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 03:47:00 pm
Last Modified: Monday, 22nd November 2021 05:21:51 pm
'''
from warnings import warn

import numpy as np
from obspy import Stream, Trace


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
        st = Stream([st])
    for ii, _ in enumerate(st):
        try:
            st[ii] = cos_taper(st[ii], taper_len, taper_at_masked)
        except ValueError as e:
            warn('%s, corresponding trace not tapered.' % e)
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
            tr.data = st[0].data
            return tr
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


def detrend_st(st: Stream, *args, **kwargs) -> Stream:
    """
    Detrends a stream while dealing with data gaps

    :param st: input Stream
    :type st: Stream
    :return: the obspy Stream detrended
    :rtype: Stream

    .. note::
        This action is performed in place. If you want to keep the
        original data use :func:`~obspy.core.stream.Stream.copy`.

    .. seealso:
        For accepted parameters consult the documentation of
        :func:`obspy.core.trace.Trace.detrend`
    """
    for tr in st:
        sst = tr.split()
        sst.detrend(*args, **kwargs)
        tr.data = sst.merge()[0].data
    return st


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
    if isinstance(st, Trace):
        st = Stream([st])
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
