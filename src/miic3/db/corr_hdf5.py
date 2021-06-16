'''
Manages the file format and class for correlations.

:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 16th April 2021 03:21:30 pm
Last Modified: Wednesday, 16th June 2021 02:52:30 pm
'''
import fnmatch
import os
import re
from typing import List
import warnings

import numpy as np
# from numpy.core.fromnumeric import compress
from obspy.core.utcdatetime import UTCDateTime
from obspy.core import Stats
import h5py

from miic3.correlate.stream import CorrStream, CorrTrace

hierarchy = "/{tag}/{network}/{station}/{channel}/{corr_st}/{corr_et}"
h5_FMTSTR = os.path.join("{dir}", "{network}.{station}.h5")


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
            if self.compression != 'gzip':
                raise ValueError(
                    'Compression of type %s is not supported.'
                    % self.compression)
            self.compression_opts = int(
                re.findall(r'(\w+?)(\d+)', compression)[0][1])
            if self.compression_opts not in np.arange(1, 10, 1, dtype=int):
                ii = np.argmin(abs(
                    np.arange(1, 10, 1, dtype=int) - self.compression_opts))
                self.compression_opts = np.arange(1, 10, 1, dtype=int)[ii]
                warnings.warn(
                    'Chosen compression level is not available for %s. \
%s Has been chosen instead (closest)' % (
                        self.compression, str(self.compression_opts)))
        else:
            self.compression = None
            self.compression_opts = None

    def _close(self):
        self.close()

    def add_correlation(
            self, data: CorrTrace or CorrStream, tag='subdivision'):
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
                tag=tag,
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
        self, network: str, station: str, channel: str, tag: str,
        corr_start: UTCDateTime = None,
            corr_end: UTCDateTime = None) -> CorrStream:
        """
        Returns a :class:`~miic3.correlate.correlate.CorrStream` holding
        all the requested data.

        ...note: **Wildcards are allowed for all parameters.**


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
            tag=tag, network=network, station=station, channel=channel,
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
                            path = tag
                        else:
                            path = '/'.join(path.split('/')[:-4])
                    else:
                        path = '/'.join(path.split('/')[:-3])
                else:
                    path = '/'.join(path.split('/')[:-2])
            else:
                path = '/'.join(path.split('/')[:-1])
        return all_traces_recursive(self[path], CorrStream(), pattern)

    def get_available_starttimes(
        self, network: str, station: str, tag: str,
            channel: str or list = '*') -> dict:
        """
        Returns a dictionary with channel codes as keys and available
        correlation starttimes as values.

        :param network: Network code (combined code, e.g., IU-YP)
        :type network: str
        :param station: Station code (combined)
        :type station: str
        :param tag: Tag
        :type tag: str
        :param channel: Channel code (combined), wildcards allowed,
            defaults to '*'
        :type channel: str, optional
        :return: A dictionary holding the availabe starttimes for each channel
            combination
        :rtype: dict
        """
        path = hierarchy.format(
            tag=tag, network=network, station=station, channel=channel,
            corr_st='*', corr_et='*')
        out = {}
        if isinstance(channel, str):
            if '*' not in channel:
                path = '/'.join(path.split('/')[:-2])
                try:
                    out[channel] = list(self[path].keys())
                except KeyError:
                    pass
                return out
            channel = [channel]
        path = '/'.join(path.split('/')[:-3])
        for ch in channel:
            for match in fnmatch.filter(self[path].keys(), ch):
                out[match] = list(self['/'.join([path, match])].keys())
        return out

    def get_available_channels(
            self, tag: str, network: str, station: str) -> List[str]:
        path = hierarchy.format(
            tag=tag, network=network, station=station, channel='*',
            corr_st='*', corr_et='*')
        path = path.split('*')[0]
        return list(self[path].keys())


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

    def __enter__(self) -> DBHandler:
        self.db_handler = DBHandler(self.path, self.mode, self.compression)
        return self.db_handler

    def __exit__(self, exc_type, exc_value, tb) -> None or bool:
        self.db_handler._close()
        if exc_type is not None:
            return False


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


def convert_header_to_hdf5(dataset: h5py.Dataset, header: Stats):
    """
    Convert an :class:`~obspy.core.Stats` object and adds it to the provided
    hdf5 dataset.

    :param dataset: the dataset that the header should be added to
    :type dataset: h5py.Dataset
    :param header: The trace's header
    :type header: Stats
    """
    header = dict(header)
    for key in header:
        try:
            if isinstance(header[key], UTCDateTime):
                # convert time to string
                header[key] = header[key].format_fissures()
            dataset.attrs[key] = header[key]
        except TypeError:
            warnings.warn(
                'The header contains an item of type %s. Information\
            of this type cannot be written to an hdf5 file.'
                % str(type(header[key])), UserWarning)
            continue


def read_hdf5_header(dataset: h5py.Dataset) -> Stats:
    """
    Takes an hdft5 dataset as input and returns the header of the CorrTrace.

    :param dataset: The dataset to be read from
    :type dataset: h5py.Dataset
    :return: The trace's header
    :rtype: Stats
    """
    attrs = dataset.attrs
    time_keys = ['starttime', 'endtime', 'corr_start', 'corr_end']
    header = {}
    for key in attrs:
        if key in time_keys:
            header[key] = UTCDateTime(attrs[key])
        elif key == 'processing':
            header[key] = list(attrs[key])
        else:
            header[key] = attrs[key]
    return Stats(header)

