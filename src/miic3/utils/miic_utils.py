'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 29th March 2021 12:54:05 pm
Last Modified: Monday, 29th March 2021 12:54:24 pm
'''

def trace_calc_az_baz_dist(tr1, tr2):
    """ Return azimut, back azimut and distance between tr1 and tr2
    This funtions calculates the azimut, back azimut and distance between tr1
    and tr2 if both have geo information in their stats dictonary.
    Required fields are:
        tr.stats.sac.stla
        tr.stats.sac.stlo

    :type tr1: :class:`~obspy.core.trace.Trace`
    :param tr1: First trace to account
    :type tr2: :class:`~obspy.core.trace.Trace`
    :param tr2: Second trace to account

    :rtype: float
    :return: **az**: Azimut angle between tr1 and tr2
    :rtype: float
    :return: **baz**: Back azimut angle between tr1 and tr2
    :rtype: float
    :return: **dist**: Distance between tr1 and tr2
    """

    if not isinstance(tr1, Trace):
        raise TypeError("tr1 must be an obspy Trace object.")

    if not isinstance(tr2, Trace):
        raise TypeError("tr2 must be an obspy Trace object.")

    try:
        from obspy.geodetics import gps2dist_azimuth 
    except ImportError:
        print("Missed obspy funciton gps2dist_azimuth")
        print("Update obspy.")
        return

    dist, az, baz = gps2dist_azimuth(tr1.stats.sac.stla, \
                                     tr1.stats.sac.stlo, \
                                     tr2.stats.sac.stla, \
                                     tr2.stats.sac.stlo)

    return az, baz, dist