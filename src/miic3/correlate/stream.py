'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th April 2021 04:19:35 pm
Last Modified: Monday, 3rd May 2021 04:47:58 pm
'''
import numpy as np
from obspy import Stream, Trace, Inventory
from obspy.core import Stats

from miic3.utils.miic_utils import trace_calc_az_baz_dist, inv_calc_az_baz_dist


class CorrStream(Stream):
    """
    Baseclass to hold correlation traces. Basically just a list of the
    correlation traces.
    """
    def __init__(self, traces: list = None):
        self.traces = []
        if isinstance(traces, CorrTrace):
            traces = [traces]
        if traces:
            for tr in traces:
                if not isinstance(tr, CorrTrace):
                    raise TypeError('Traces have to be of type \
                        :class:`~miic3.correlate.correlate.CorrTrace`.')
                self.traces.append(tr)


class CorrTrace(Trace):
    """
    Baseclass to hold correlation data. Derived from the class
    :class:`~obspy.core.trace.Trace`.
    """
    def __init__(
        self, data: np.ndarray, header1: Stats = None,
        header2: Stats = None, inv: Inventory = None,
        start_lag: float = None, end_lag: float = None,
            _header: dict = None):
        """
        Initialise the correlation trace. Is done by combining the stats of the
        two :class:`~obspy.core.trace.Trace` objects' headers. If said headers
        do not contain Station information (i.e., coordinates), an
        :class:`~obspy.core.inventory.Inventory` with information about both
        stations should be provided as well.

        :param data: The correlation data
        :type data: np.ndarray
        :param header1: header of the first trace, defaults to None
        :type header1: Stats, optional
        :param header2: header of the second trace, defaults to None
        :type header2: Stats, optional
        :param inv: Inventory object for the stations, defaults to None
        :type inv: Inventory, optional
        :param start_lag: The lag of the first sample of the correlation given
        in seconds.
        :type start_lag: float
        :param end_lag: The lag of the last sample of the correlation
        in seconds.
        :type end_lag: float
        :param _header: Already combined header, used when reading correlations
            from a file, defaults to None
        :type _header: dict, optional
        """
        if _header:
            header = Stats(_header)
        elif not header1 and not header2:
            header = Stats()
            if start_lag and end_lag:
                header['start_lag'] = start_lag
                header['end_lag'] = end_lag
        else:
            # make sure the order is correct
            # Will do that always alphabetically sorted
            sort1 = header1.network + header1.station + header1.channel
            sort2 = header2.network + header2.station + header2.channel
            sort = [sort1, sort2]
            sorted = sort.copy()
            sorted.sort()
            if sort != sorted:
                header = combine_stats(
                    header2, header1, end_lag,
                    start_lag, inv=inv)
                # reverse array and lag times
                data = np.flip(data)
            else:
                header = combine_stats(
                    header1, header2, start_lag,
                    end_lag, inv=inv)
        header['npts'] = len(data)
        super(CorrTrace, self).__init__(data=data, header=header)
        # st = self.stats
        # if ('_format' in st and st._format.upper() == 'Q' and
        #         st.station.count('.') > 0):
        #     st.network, st.station, st.location = st.station.split('.')[:3]
        # self._read_format_specific_header()


def combine_stats(
    stats1: Stats, stats2: Stats, start_lag: float,
        end_lag: float, inv: Inventory = None):
    """ Combine the meta-information of two ObsPy Trace.Stats objects

    This function returns a ObsPy :class:`~obspy.core.trace.Stats` object
    obtained combining the two associated with the input Traces.
    Namely ``stats1`` and ``stats2``.

    The fields ['network','station','location','channel'] are combined in
    a ``-`` separated fashion to create a "pseudo" SEED like ``id``.

    For all the others fields, only "common" information are retained: This
    means that only keywords that exist in both dictionaries will be included
    in the resulting one.

    :type stats1: :class:`~obspy.core.trace.Stats`
    :param stats1: First Trace's stats
    :type stats2: :class:`~obspy.core.trace.Stats`
    :param stats2: Second Trace's stats
    :param start_lag: The lag of the first sample of the correlation given
    in seconds.
    :type start_lag: float
    :param end_lag: The lag of the last sample of the correlation given
    in seconds.
    :type end_lag: float
    :type inv: :class:`~obspy.core.inventory.Inventory`, optional
    :param inv: Inventory containing the station coordinates. Only needed if
        station coordinates are not in Trace.Stats. Defaults to None.

    :rtype: :class:`~obspy.core.trace.Stats`
    :return: **stats**: combined Stats object
    """

    if not isinstance(stats1, Stats):
        raise TypeError("stats1 must be an obspy Stats object.")

    if not isinstance(stats2, Stats):
        raise TypeError("stats2 must be an obspy Stats object.")

    # We also have to remove these as they are obspy AttributeDicts as well
    stats1.pop('asdf', None)
    stats2.pop('asdf', None)

    tr1_keys = list(stats1.keys())
    tr2_keys = list(stats2.keys())

    stats = Stats()
    # actual correlation times
    stats['corr_start'] = max(stats1.starttime, stats2.starttime)
    stats['corr_end'] = min(stats1.endtime, stats2.endtime)
    # This makes stats['endtime'] meaningsless, but obspy needs something that
    # identifies the Trace as unique
    stats['startime'] = stats['corr_start']

    # Adjust the information to create a new SEED like id
    keywords = ['network', 'station', 'location', 'channel']
    sac_keywords = ['sac']

    for key in keywords:
        if key in tr1_keys and key in tr2_keys:
            stats[key] = stats1[key] + '-' + stats2[key]

    for key in tr1_keys:
        if key not in keywords and key not in sac_keywords:
            if key in tr2_keys:
                if stats1[key] == stats2[key]:
                    # in the stats object there are read only objects
                    try:
                        stats[key] = stats1[key]
                    except AttributeError:
                        pass

    try:
        stats['stla'] = stats1.sac.stla
        stats['stlo'] = stats1.sac.stlo
        stats['stel'] = stats1.sac.stel
        stats['evla'] = stats2.sac.stla
        stats['evlo'] = stats2.sac.stlo
        stats['evel'] = stats2.sac.stel

        az, baz, dist = trace_calc_az_baz_dist(stats1, stats2)

        stats['dist'] = dist / 1000
        stats['az'] = az
        stats['baz'] = baz
    except AttributeError:
        if inv:
            inv1 = inv.select(
                network=stats1.network, station=stats1.station)
            inv2 = inv.select(
                network=stats2.network, station=stats2.station)
            stats['stla'] = inv1[0][0].latitude
            stats['stlo'] = inv1[0][0].longitude
            stats['stel'] = inv1[0][0].elevation
            stats['evla'] = inv2[0][0].latitude
            stats['evlo'] = inv2[0][0].longitude
            stats['evel'] = inv2[0][0].elevation

            az, baz, dist = inv_calc_az_baz_dist(inv1, inv2)

            stats['dist'] = dist / 1000
            stats['az'] = az
            stats['baz'] = baz
        else:
            print("No station coordinates provided.")
    stats.pop('sac', None)
    stats.pop('response', None)
    stats['_format'] = 'hdf5'

    # note that those have to be adapted whenever several correlations are
    # stacked
    stats['start_lag'] = start_lag
    stats['end_lag'] = end_lag
    return stats
