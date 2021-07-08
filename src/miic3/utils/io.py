'''
:copyright:
:license:
   `GNU Lesser General Public License, Version 3 <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 8th July 2021 10:37:21 am
Last Modified: Thursday, 8th July 2021 04:04:52 pm
'''
from collections import Iterable
import os
from warnings import warn

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Stats
from scipy.io import loadmat
from miic3.correlate.stats import CorrStats

from miic3.correlate.stream import CorrTrace, CorrStream,\
    CorrBulk


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


def load_corrbulk_from_mat(
        dir: str, net: str, station: str, channel: str) -> CorrBulk:
    ctrl = []  # list of correlation traces
    for path, _, fi in os.walk(dir):
        for f in fi:
            if (net and station and channel) not in f:
                continue
            try:
                ctrl.append(mat_to_corrtrace(loadmat(os.path.join(path, f))))
            except KeyError:
                warn('No data for %s/%s' % (path, f))
                continue
    if not len(ctrl):
        raise FileNotFoundError('No Correlations found.')
    cst = CorrStream(ctrl)
    return cst.create_corr_bulk()


def corrmat_to_corrbulk(path: str) -> CorrBulk:
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
