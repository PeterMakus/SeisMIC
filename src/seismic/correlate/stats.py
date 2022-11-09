'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 5th July 2021 02:44:13 pm
Last Modified: Thursday, 3rd November 2022 10:54:02 am
'''
from obspy.core.util import AttribDict
from obspy import UTCDateTime


class CorrStats(AttribDict):
    """
    From Obspy, but with the difference that some items can be lists and that
    corr_start and corr_end are introduced, some other differences only for
    correlations.

    A container for additional header information of a ObsPy Trace object.

    A ``Stats`` object may contain all header information (also known as meta
    data) of a :class:`~seismic.correlate.stream.CorrTrace` object. Those
    headers may be accessed or modified either in the dictionary style or
    directly via a corresponding attribute. There are various default
    attributes which are required by every waveform import and export modules
    within ObsPy such as :mod:`obspy.io.mseed`.

    :type header: dict or :class:`~obspy.core.trace.Stats`, optional
    :param header: Dictionary containing meta information of a single
        :class:`~obspy.core.trace.Trace` object. Possible keywords are
        summarized in the following `Default Attributes`_ section.

    .. rubric:: Basic Usage

    >>> stats = CorrStats()
    >>> stats.network = 'BW'
    >>> print(stats['network'])
    BW
    >>> stats['station'] = 'MANZ'
    >>> print(stats.station)
    MANZ

    .. rubric:: _`Default Attributes`

    ``sampling_rate`` : float, optional
        Sampling rate in hertz (default value is 1.0).
    ``delta`` : float, optional
        Sample distance in seconds (default value is 1.0).
    ``calib`` : float, optional
        Calibration factor (default value is 1.0).
    ``npts`` : int, optional
        Number of sample points (default value is 0, which implies that no data
        is present).
    ``network`` : string, optional
        Network code (default is an empty string).
    ``location`` : string, optional
        Location code (default is an empty string).
    ``station`` : string, optional
        Station code (default is an empty string).
    ``channel`` : string, optional
        Channel code (default is an empty string).
    ``starttime`` : :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Date and time of the first data sample used for correlation in UTC
        (default value is "1970-01-01T00:00:00.0Z").
    ``endtime`` : :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Date and time of the last data sample used for correlation in UTC
        (default value is "1970-01-01T00:00:00.0Z").
    ``corr_start``: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Date and time of the first data sample used for correlation in UTC
        (default value is "1970-01-01T00:00:00.0Z").
    ``corr_end``: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Date and time of the last data sample used for correlation in UTC
        (default value is "1970-01-01T00:00:00.0Z").
    ``start_lag``: float, optional
        Lag of the first sample in seconds (usually negative)
    ``end_lag``: float, optional
        Lag of the last sample in seconds.

    .. rubric:: Notes

    (1) The attributes ``sampling_rate`` and ``delta`` are linked to each
        other. If one of the attributes is modified the other will be
        recalculated.

        >>> stats = Stats()
        >>> stats.sampling_rate
        1.0
        >>> stats.delta = 0.005
        >>> stats.sampling_rate
        200.0

    (2) The attributes ``start_lag``, ``npts``, ``sampling_rate`` and ``delta``
        are monitored and used to automatically calculate the ``end_lag``.

        >>> stats = Stats()
        >>> stats.npts = 61
        >>> stats.delta = 1.0
        >>> stats.start_lag = -30
        >>> stats.end_lag
        30
        >>> stats.delta = 0.5
        >>> stats.end_lag
        0

    (3) The attribute ``endtime``, ``end_lag``, and ``starttime`` are
        read only and cannot be modified. ``starttime`` and ``endtime`` are
        just simply aliases of ``corr_start`` and ``corr_end``.

        >>> stats = Stats()
        >>> stats.endtime = UTCDateTime(2009, 1, 1, 12, 0, 0)
        Traceback (most recent call last):
        ...
        AttributeError: Attribute "endtime" in Stats object is read only!
        >>> stats['endtime'] = UTCDateTime(2009, 1, 1, 12, 0, 0)
        Traceback (most recent call last):
        ...
        AttributeError: Attribute "endtime" in Stats object is read only!

    (4)
        The attribute ``npts`` will be automatically updated from the
        :class:`~seismic.correlate.stream.CorrTrace` object.

        >>> trace = CorrTrace()
        >>> trace.stats.npts
        0
        >>> trace.data = np.array([1, 2, 3, 4])
        >>> trace.stats.npts
        4

    (5)
        The attribute ``component`` can be used to get or set the component,
        i.e. the last character of the ``channel`` attribute.

        >>> stats = Stats()
        >>> stats.channel = 'HHZ'
        >>> stats.component  # doctest: +SKIP
        'Z'
        >>> stats.component = 'L'
        >>> stats.channel  # doctest: +SKIP
        'HHL'

    """
    # set of read only attrs
    readonly = ['endtime', 'end_lag', 'starttime']
    # default values
    defaults = {
        'sampling_rate': 1.0,
        'delta': 1.0,
        'starttime': UTCDateTime(0),
        'endtime': UTCDateTime(0),
        'corr_start': UTCDateTime(0),
        'corr_end': UTCDateTime(0),
        'start_lag': 0,
        'end_lag': 0,
        'npts': 0,
        'calib': 1.0,
        'network': '',
        'station': '',
        'location': '',
        'channel': '',
    }
    # keys which need to refresh derived values
    _refresh_keys = {
        'delta', 'sampling_rate', 'corr_start', 'corr_end', 'npts',
        'start_lag'}
    # dict of required types for certain attrs
    _types = {
        'network': (str),
        'station': (str),
    }

    def __init__(self, header={}):
        """
        """
        super(CorrStats, self).__init__(header)

    def __setitem__(self, key, value):
        """
        """
        if key in self._refresh_keys:
            # ensure correct data type
            if key == 'delta':
                key = 'sampling_rate'
                try:
                    value = 1.0 / float(value)
                except ZeroDivisionError:
                    value = 0.0
            elif key == 'sampling_rate':
                value = float(value)
            elif key == 'start_lag':
                value = float(value)
            elif key == 'npts':
                if not isinstance(value, int):
                    value = int(value)
            # set current key
            super(CorrStats, self).__setitem__(key, value)
            # set derived value: delta
            try:
                delta = 1.0 / float(self.sampling_rate)
            except ZeroDivisionError:
                delta = 0
            self.__dict__['delta'] = delta
            # set derived value: endtime
            if self.npts == 0:
                timediff = 0
            else:
                timediff = float(self.npts - 1) * delta
            self.__dict__['end_lag'] = self.start_lag + timediff
            self.__dict__['endtime'] = self.corr_end
            self.__dict__['starttime'] = self.corr_start
            return
        if key == 'component':
            key = 'channel'
            value = str(value)
            if len(value) != 3:
                msg = 'Component must be set with three characters, e.g. E-Z'
                raise ValueError(msg)
            a, b = self.channel.split('-')
            an, bn = value.split('-')
            value = a[:-1] + an + '-' + b[:-1] + bn
        # all other keys
        if isinstance(value, dict):
            super(CorrStats, self).__setitem__(key, AttribDict(value))
        else:
            super(CorrStats, self).__setitem__(key, value)

    __setattr__ = __setitem__

    def __getitem__(self, key, default=None):
        """
        """
        if key == 'component':
            ch = super(CorrStats, self).__getitem__('channel', default)
            a, b = ch.split('-')
            return a[-1]+'-'+b[-1]
        else:
            return super(CorrStats, self).__getitem__(key, default)

    def __str__(self):
        """
        Return better readable string representation of Stats object.
        """
        priorized_keys = ['network', 'station', 'location', 'channel',
                          'corr_start', 'corr_end', 'start_lag', 'end_lag',
                          'sampling_rate', 'delta', 'npts', 'calib']
        return self._pretty_str(priorized_keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
