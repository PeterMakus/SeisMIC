'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 29th March 2021 12:54:05 pm
Last Modified: Tuesday, 18th May 2021 11:01:08 am
'''
from obspy import Inventory, Stream
from obspy.core import Stats


def trace_calc_az_baz_dist(stats1: Stats, stats2: Stats):
    """ Return azimuth, back azimhut and distance between tr1 and tr2
    This funtions calculates the azimut, back azimut and distance between tr1
    and tr2 if both have geo information in their stats dictonary.
    Required fields are:
        tr.stats.sac.stla
        tr.stats.sac.stlo

    :type stats1: :class:`~obspy.core.Stats`
    :param stats1: First trace to account
    :type stats2: :class:`~obspy.core.Stats`
    :param stats2: Second trace to account

    :rtype: float
    :return: **az**: Azimuth angle between tr1 and tr2
    :rtype: float
    :return: **baz**: Back-azimuth angle between tr1 and tr2
    :rtype: float
    :return: **dist**: Distance between tr1 and tr2
    """

    if not isinstance(stats1, Stats):
        raise TypeError("stats1 must be an obspy Stats object.")

    if not isinstance(stats2, Stats):
        raise TypeError("stats2 must be an obspy Stats object.")

    try:
        from obspy.geodetics import gps2dist_azimuth
    except ImportError:
        print("Missed obspy funciton gps2dist_azimuth")
        print("Update obspy.")
        return

    dist, az, baz = gps2dist_azimuth(stats1.sac.stla,
                                     stats1.sac.stlo,
                                     stats2.sac.stla,
                                     stats2.sac.stlo)

    return az, baz, dist


def inv_calc_az_baz_dist(inv1: Inventory, inv2: Inventory):
    """ Return azimuth, back azimuth and distance between stat1 and stat2


    :type tr1: :class:`~obspy.core.inventory.Inventory`
    :param tr1: First trace to account
    :type tr2: :class:`~obspy.core.inventory.Inventory`
    :param tr2: Second trace to account

    :rtype: float
    :return: **az**: Azimuth angle between stat1 and stat2
    :rtype: float
    :return: **baz**: Back-azimuth angle between stat2 and stat2
    :rtype: float
    :return: **dist**: Distance between stat1 and stat2
    """

    if not isinstance(inv1, Inventory):
        raise TypeError("inv1 must be an obspy Inventory.")

    if not isinstance(inv2, Inventory):
        raise TypeError("inv2 must be an obspy Inventory.")

    try:
        from obspy.geodetics import gps2dist_azimuth
    except ImportError:
        print("Missing obspy funciton gps2dist_azimuth")
        print("Update obspy.")
        return

    dist, az, baz = gps2dist_azimuth(inv1[0][0].latitude,
                                     inv1[0][0].longitude,
                                     inv2[0][0].latitude,
                                     inv2[0][0].longitude)

    return az, baz, dist


def resample_or_decimate(
        data: Stream, sampling_rate_new: int, filter=True) -> Stream:
    """
    Decimates the data if the desired new sampling rate allows to do so.
    Else the signal will be interpolated (a lot slower).

    :param data: Stream to be resampled.
    :type data: Stream
    :param sampling_rate_new: The desired new sampling rate
    :type sampling_rate_new: int
    :return: The resampled stream
    :rtype: Stream
    """
    sr = data[0].stats.sampling_rate
    srn = sampling_rate_new

    # Chosen this filter design as it's exactly the same as
    # obspy.Stream.decimate uses
    if filter:
        freq = sr * 0.5 / float(sr/srn)
        data.filter('lowpass_cheby_2', freq=freq, maxorder=12)

    if sr/srn == sr//srn:
        return data.decimate(int(sr//srn), no_filter=True)
    else:
        return data.resample(srn)
