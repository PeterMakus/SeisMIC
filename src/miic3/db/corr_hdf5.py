'''
Manages the file format and class for correlations.

:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 16th April 2021 03:21:30 pm
Last Modified: Wednesday, 28th April 2021 03:53:51 pm
'''
import fnmatch
import os
import re
import warnings

import numpy as np
# from numpy.core.fromnumeric import compress
from obspy.core.utcdatetime import UTCDateTime
from obspy.core import Stats
import h5py

from miic3.correlate.stream import CorrStream, CorrTrace

base_path = '/corr_data'
hierarchy = base_path + "/{network}/{station}/{channel}/{corr_st}/{corr_et}"
h5_FMTSTR = os.path.join("{dir}", "{network}.{station}.h5")


class CorrelationDataBase(object):
    """
    Base class to handle the hdf5 files that contain noise correlations.
    """
    def __init__(self, path: str, mode: str = 'a', compression: str = 'gzip3'):
        """
        Access an hdf5 file holding correlations.

        :warning: **Only access through a context manager (see below):**

        >>> with CorrelationDataBase(myfile.h5) as cdb:
        >>>     type(cdb)  # This is a DBHandler

        :param path: Full path to the file
        :type path: str
        :param mode: Mode to access the file. Options are: 'a' for all, 'w' for
        write, 'r+' for writing in an already existing file, or 'r' for
        read-only , defaults to 'a'.
        :type mode: str, optional
        :param compression: The compression algorithm and compression level
        that the arrays should be saved with. 'gzip3' tends to perform well,
        else you could choose 'gzipx' where x is a digit between 1 and 9 (i.e.,
        9 is the highest compression) or None for fastest perfomance,
        defaults to 'gzip3'.
        :type compression: str, optional
        """
        # Create / read file
        if not path.split('.')[-1] == 'h5':
            path += '.h5'
        self.path = path
        self.mode = mode
        self.compression = compression

    def __enter__(self):
        self.db_handler = DBHandler(self.path, self.mode, self.compression)
        return self.db_handler

    def __exit__(self, exc_type, exc_value, tb):
        self.db_handler._close()
        if exc_type is not None:
            return False


class DBHandler(h5py.File):
    """
    The actual file handler of the hdf5 correlation files.

    :note: **Should not be accessed directly. Access
    :class:`~miic3.db.corr_hdf5.CorrelationDataBase` instead.**

    Child object of :class:`h5py.File` and inherets all its attributes and
    functions in addition to functions that are particularly useful for noise
    correlations.
    """
    def __init__(self, path, mode, compression):
        super(DBHandler, self).__init__(path, mode=mode)
        if isinstance(compression, str):
            self.compression = re.findall(r'(\w+?)(\d+)', compression)[0][0]
            self.compression_opts = int(
                re.findall(r'(\w+?)(\d+)', compression)[0][1])
        else:
            self.compression = None
            self.compression_opts = None

    def _close(self):
        self.close()

    def add_correlation(self, data: CorrTrace or CorrStream):
        """
        Add correlation data to the hdf5 file. Can be accessed using the
        :func:`~miic3.db.corr_hdf5.DBHandler.get_data()` method.

        :param data: Data to save. Either a
        :class:`miic3.correlate.correlate.CorrTrace` object or a
        :class:`miic3.correlate.correlate.CorrStream` holding one or several
        traces.
        :type data: CorrTraceorCorrStream
        :raises TypeError: for wrong data type.
        """
        # There should be a CorrTrace and CorrStream object that are subclasses
        # of the obspy classes that can be used here.
        if not isinstance(data, CorrTrace) and\
                not isinstance(data, CorrStream):
            raise TypeError('Data has to be either a \
            :class:`~miic3.correlate.correlate.CorrTrace` object or a \
            :class:`~miic3.correlate.correlate.CorrStream` object')

        if isinstance(data, CorrTrace):
            data = [data]

        for tr in data:
            st = tr.stats
            path = hierarchy.format(
                network=st.network, station=st.station, channel=st.channel,
                corr_st=st.corr_start.format_fissures(),
                corr_et=st.corr_end.format_fissures())
            try:
                ds = self.create_dataset(
                    path, data=tr.data, compression=self.compression,
                    compression_opts=self.compression_opts)
                convert_header_to_hdf5(ds, st)
            except ValueError as e:
                print(e)
                warnings.warn("The dataset %s is already in file and will be \
                omitted." % path, category=UserWarning)

    def get_data(
        self, network: str, station: str, channel: str,
        corr_start: UTCDateTime = None,
            corr_end: UTCDateTime = None) -> CorrStream:
        """
        Returns a :class:`~miic3.correlate.correlate.CorrStream` holding
        all the requested data. **Wildcards are allowed for all parameters.**


        :param network: network (combination), e.g., IU-YP
        :type network: str
        :param station: station (combination), e.g., HRV-BRK
        :type station: str
        :param channel: channel (combination), e.g., BZ-BR
        :type channel: str
        :param corr_start: starttime of the time windows used to computed this
        correlation, defaults to None
        :type corr_start: UTCDateTime, optional
        :param corr_end: endtime of the time windows used to computed this
        correlation, defaults to None
        :type corr_end: UTCDateTime, optional
        :return: a :class:`~miic3.correlate.correlate.CorrStream` holding
        all the requested data.
        :rtype: CorrStream
        """
        # Make sure the request is structured correctly
        sort = ['.'.join([net, stat, chan]) for net, stat, chan in zip(
                network.split('-'), station.split('-'), channel.split('-'))]
        sorted = sort.copy()
        sorted.sort()
        if sorted != sort:
            network, station, channel = ['-'.join([a, b]) for a, b in zip(
                sorted[0].split('.'), sorted[1].split('.'))]

        if isinstance(corr_start, UTCDateTime):
            corr_start = corr_start.format_fissures()
        else:
            corr_start = '*'
        if isinstance(corr_end, UTCDateTime):
            corr_end = corr_end.format_fissures()
        else:
            corr_end = '*'
        path = hierarchy.format(
            network=network, station=station, channel=channel,
            corr_st=corr_start, corr_et=corr_end)
        # Extremely ugly way of changing the path
        if '*' not in path:
            data = np.array(self[path])
            header = read_hdf5_header(self[path])
            return CorrStream(CorrTrace(data, _header=header))
        # Now, we need to differ between the fnmatch pattern and the actually
        # acessed path
        pattern = path.replace('/*', '*')
        if corr_end == '*':
            if corr_start == '*':
                if channel == '*':
                    if station == '*':
                        if network == '*':
                            path = base_path
                        else:
                            path = '/'.join(path.split('/')[:-4])
                    else:
                        path = '/'.join(path.split('/')[:-3])
                else:
                    path = '/'.join(path.split('/')[:-2])
            else:
                path = '/'.join(path.split('/')[:-1])
        return all_traces_recursive(self[path], CorrStream(), pattern)


def all_traces_recursive(
    group: h5py._hl.group.Group, stream: CorrStream,
        pattern: str) -> CorrStream:
    """
    Recursively, appends all traces in a h5py group to the input stream.
    In addition this will check whether the data matches a certain pattern.

    :param group: group to search through
    :type group: class:`h5py._hl.group.Group`
    :param stream: Stream to append the traces to
    :type stream: CorrStream
    :param pattern: pattern for the path in the hdf5 file, see fnmatch for
        details.
    :type pattern: str
    :return: Stream with appended traces
    :rtype: CorrStream
    """
    for v in group.values():
        if not fnmatch.fnmatch(v.name, pattern) and v.name not in pattern:
            continue
        if isinstance(v, h5py._hl.group.Group):
            all_traces_recursive(v, stream, pattern)
        else:
            stream.append(CorrTrace(np.array(v), _header=read_hdf5_header(v)))
    return stream


def convert_header_to_hdf5(dataset, header):
    header = dict(header)
    for key in header:
        if isinstance(header[key], UTCDateTime):
            # convert time to string
            header[key] = header[key].format_fissures()
        try:
            dataset.attrs[key] = header[key]
        except TypeError:
            warnings.warn(
                'The header contains an item of type %s. Information\
             of this type cannot be written to an hdf5 file.'
                % str(type(header[key])), UserWarning)
            continue


def read_hdf5_header(dataset) -> Stats:
    attrs = dataset.attrs
    time_keys = ['starttime', 'endtime', 'start_corr_time', 'end_corr_time']
    header = {}
    for key in attrs:
        if key in time_keys:
            header[key] = UTCDateTime(attrs[key])
        else:
            header[key] = attrs[key]
    return Stats(header)
