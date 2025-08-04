'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    `EUROPEAN UNION PUBLIC LICENCE v. 1.2
    <https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>`_
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 15th November 2022 05:15:30 pm
Last Modified: Wednesday, 25th Febuary 2025 01:48:00 pm (J. Lehr)
'''

from typing import Tuple

import numpy as np

from seismic.correlate.stats import CorrStats

from .. import logfactory
import logging

parentlogger = logfactory.create_logger()
module_logger = logging.getLogger(parentlogger.name + '.trim')


def corr_mat_trim(
    data: np.ndarray, stats: CorrStats, starttime: float,
        endtime: float) -> Tuple[np.ndarray, CorrStats]:
    """ Trim the correlation matrix to a given period.

    Trim the correlation matrix `corr_mat` to the period from `starttime` to
    `endtime` given in seconds from the zero position, so both can be
    positive and negative. If `starttime` and `endtime` are datetime.datetime
    objects they are taken as absolute start and endtime.

    :type data: np.ndarray
    :param corr_mat: correlation matrix to be trimmed
    :type starttime: float
    :param starttime: start time in seconds with respect to the zero position.
        Hence, this parameter describes a lag!
    :type endtime: float
    :param order: end time in seconds with respect to the zero position

    :rtype tuple: Tuple[np.ndarray, CorrStats]
    :return: trimmed correlation matrix and new stats object
    """

    # fetch indices of start and end
    start = int(
        np.floor((starttime-stats['start_lag'])*stats['sampling_rate']))
    end = int(np.floor((endtime-stats['start_lag'])*stats['sampling_rate']))

    # check range
    if start < 0:
        module_logger.warning(
            'Error: starttime before beginning of trace. Data not changed')
        return data, stats
    if end >= stats['npts']:
        module_logger.warning(
            'Error: endtime after end of trace. Data not changed')
        return data, stats

    # select requested part from matrix
    # +1 is to include the last sample
    if len(data.shape) == 1:
        data = data[start:end+1]
        stats['npts'] = len(data)
    else:
        data = data[:, start:end + 1]
        stats['npts'] = data.shape[1]

    # set starttime, endtime and npts of the new stats
    stats['start_lag'] = starttime

    return data, stats
