'''
:copyright:
:license:
   `GNU Lesser General Public License, Version 3 <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 8th July 2021 10:37:21 am
Last Modified: Tuesday, 13th July 2021 03:01:51 pm
'''
from collections import Iterable

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Stats
from scipy.io import loadmat
from miic3.correlate.stats import CorrStats

from miic3.correlate.stream import CorrTrace, CorrBulk


def flatten(x):
    """ Return the flattened version of the input array x

    This funciton works with all :class:`~collections.Iterable` that can be
    nested in an irregular fashion.

    .. rubric: Example

    >>>L=[[[1, 2, 3], [4, 5]], 6]
    >>>flatten(L)
    [1, 2, 3, 4, 5, 6]

    :type x: :class:`~collections.Iterable`
    :param: Iterable to be flattened

    :rtype: list
    :return: x as a flattened list
    """
    result = []
    for el in x:
        if isinstance(el, Iterable) and \
            not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def flatten_recarray(x):
    """ Flatten a recarray
    """
    flat_dict = {}
    if hasattr(x, 'dtype'):
        if hasattr(x.dtype, 'names'):
            if x.dtype.names is not None:
                for name in x.dtype.names:
                    field = flatten(x[name])
                    if field != []:
                        flat_dict.update({name: field[0]})
                    else:
                        flat_dict.update({name: field})
                return flat_dict
    return x


zerotime = UTCDateTime(1971, 1, 1)


def mat_to_corrtrace(mat: dict) -> CorrTrace:
    """
    Reads in a correlation dictionary created with the old (i.e., python2)
    miic and translates it into a :class:`~miic3.correlate.stream.CorrTrace`
    object.

    :param mat: dictionary as produced by :func:`scipy.io.loadmat`
    :type mat: dict
    :return: A Correlation Trace
    :rtype: CorrTrace
    """
    for key in mat:
        if mat[key] is not None:
            flat_var = flatten_recarray(mat[key])
            mat.update({key: flat_var})
    start_lag = UTCDateTime(mat['stats']['starttime']) - zerotime
    end_lag = UTCDateTime(mat['stats']['endtime']) - zerotime
    if not isinstance(mat['stats_tr1']['location'], str):
        mat['stats_tr1'].update({'location': '-'})
    if not isinstance(mat['stats_tr2']['location'], str):
        mat['stats_tr2'].update({'location': '-'})
    ctr = CorrTrace(
        mat['corr_trace'].flatten(), Stats(mat['stats_tr1']),
        Stats(mat['stats_tr2']), start_lag=start_lag, end_lag=end_lag)
    return ctr


def corrmat_to_corrbulk(path: str) -> CorrBulk:
    """
    Reads in a correlation .mat file created with the old (i.e., python2)
    miic and translates it into a :class:`~miic3.correlate.stream.CorrTrace`
    object.

    :param path: Path to file
    :type path: str
    :return: A Correlation Bulk object
    :rtype: CorrBulk

    :warning: This reads both stacks and subdivisions into the same object.
    """
    mat = loadmat(path)
    for key in mat:
        if mat[key] is not None:
            flat_var = flatten_recarray(mat[key])
            mat.update({key: flat_var})
    stats = CorrStats()
    start_lag = UTCDateTime(mat['stats']['starttime']) - zerotime
    data = mat.pop('corr_data')
    corr_start = [UTCDateTime(t) for t in mat['time']]
    mat['stats'].pop('starttime')
    stats.update(mat['stats'])
    stats.update({'start_lag': start_lag, 'corr_start': corr_start})
    return CorrBulk(data, stats)
