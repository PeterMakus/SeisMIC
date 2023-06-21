'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 18th February 2021 02:30:02 pm
Last Modified: Wednesday, 21st June 2023 05:37:29 pm
'''

import fnmatch
import os
import datetime
import glob
from typing import List, Tuple, Iterator
import warnings
import re

import numpy as np
from obspy.clients.fdsn import Client as rClient
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.filesystem.sds import Client as lClient
from obspy import read_inventory, UTCDateTime, read, Stream, Inventory
from obspy.clients.fdsn.mass_downloader import RectangularDomain, \
    Restrictions, MassDownloader

from seismic.utils.raw_analysis import spct_series_welch


SDS_FMTSTR = os.path.join(
    "{year}", "{network}", "{station}", "{channel}.{sds_type}",
    "{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}")


class Store_Client(object):
    """
    Client for request and local storage of waveform data

    Request client that stores downloaded data for later local retrieval.
    When reading stored data the client reads from the local copy of the data.
    Inventory data is stored in the folder `inventory` and attached to data
    that is read.
    """

    def __init__(self, Client: rClient, path: str, read_only: bool = False):
        """
        Initialize the client

        :type Client: obspy request client that supports the 'get_waveform'
            method
        :param Client: obspy request client that supports the 'get_waveform'
            method
        :type path: str
        :param path: path of the store's sds root directory
        :type read_only: Bool
        :param read_only: If True the local archive is not extended.
        """
        assert os.path.isdir(path), "{} is not a directory".format(path)
        self.fileborder_seconds = 30
        self.fileborder_samples = 5000
        self.sds_root = os.path.join(path, 'mseed')
        if not read_only:
            os.makedirs(self.sds_root, exist_ok=True)
        self.inv_dir = os.path.join(path, "inventory")
        if os.path.isdir(self.inv_dir) and os.listdir(self.inv_dir):
            self.inventory = self.read_inventory()
        else:
            self.inventory = Inventory()
        if not read_only:
            os.makedirs(self.inv_dir, exist_ok=True)
        self.sds_type = "D"
        self.lclient = lClient(
            self.sds_root, sds_type=self.sds_type, format="MSEED",
            fileborder_seconds=self.fileborder_seconds,
            fileborder_samples=self.fileborder_samples)
        self.rclient = Client
        self.read_only = read_only

    def download_waveforms_mdl(
        self, starttime: UTCDateTime, endtime: UTCDateTime,
        clients: list or None = None, minlat: float or None = None,
        maxlat: float or None = None, minlon: float or None = None,
        maxlon: float or None = None, network: str or None = None,
        station: str or None = None, location: str or None = None,
            channel: str or None = None):
        # Initialise MassDownloader

        domain = RectangularDomain(
            minlatitude=minlat, maxlatitude=maxlat, minlongitude=minlon,
            maxlongitude=maxlon)

        restrictions = Restrictions(
            starttime=starttime,
            endtime=endtime,
            # Chunk it to have one file per day.
            chunklength_in_sec=86400,
            # Considering the enormous amount of data associated with
            # continuous requests, you might want to limit the data based on
            # SEED identifiers. If the location code is specified, the location
            # priority list is not used; the same is true for the channel
            # argument and priority list.
            network=network, station=station, location=location,
            channel=channel,
            # The typical use case for such a data set are noise correlations
            # where gaps are dealt with at a later stage.
            reject_channels_with_gaps=False,
            # Same is true with the minimum length. All data might be useful.
            minimum_length=0.0,
            # Guard against the same station having different names.
            minimum_interstation_distance_in_m=100.0,
            sanitize=False)

        mdl = MassDownloader(providers=clients)

        mdl.download(
            domain, restrictions, mseed_storage=self._get_mseed_storage,
            stationxml_storage=self.inv_dir)

    def get_waveforms(
        self, network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime, endtime: UTCDateTime,
            attach_response: bool = True, _check_times: bool = True) -> Stream:
        assert ~('?' in network+station+location+channel), \
            "Only single channel requests supported."
        assert ~('*' in network+station+location+channel), \
            "Only single channel requests supported."
        # try to load from local disk
        st = self._load_local(network, station, location, channel, starttime,
                              endtime, attach_response, _check_times)
        if st is None:
            st = self._load_remote(
                network, station, location, channel, starttime, endtime,
                attach_response)
        return st

    def get_available_stations(self, network: str = None) -> list:
        """
        Returns a list of stations for which raw data is available.
        Very similar to the
        :func:`~obspy.clients.filesystem.sds.get_all_stations()` method with
        the difference that this one allows to filter for particular networks.

        :param network: Only return stations from this network.
            ``network==None`` is same as a wildcard. Defaults to None.
        :type network: str or None, optional
        :return: List of network and station codes in the form:
            `[[net0,stat0], [net1,stat1]]`.
        :rtype: list
        """
        # If no network is defined use all
        network = network or '*'
        oslist = glob.glob(
            os.path.join(self.sds_root, '????', network, '*'))
        statlist = []
        for path in oslist:
            if not isinstance(eval(path.split(os.path.sep)[-3]), int):
                continue
            # Add all network and station combinations to list
            code = path.split(os.path.sep)[-2:]
            if code not in statlist:
                statlist.append(code)
        if not statlist:
            raise FileNotFoundError('No Data Available.')
        return statlist

    def _get_mseed_storage(
        self, network: str, station: str, location: str,
        channel: str, starttime: UTCDateTime,
            endtime: UTCDateTime) -> bool or str:

        # Returning True means that neither the data nor the StationXML file

        # will be downloaded.

        filename = SDS_FMTSTR.format(
            network=network, station=station, location=location,
            channel=channel, year=starttime.year, doy=starttime.julday,
            sds_type=self.sds_type)
        outf = os.path.join(self.sds_root, filename)

        if os.path.isfile(outf):
            return True

        # If a string is returned the file will be saved in that location.

        return outf

    def _get_times(self, network: str, station: str) -> Tuple[
            UTCDateTime, UTCDateTime]:
        """
        Return earliest and latest available time for a certain station.

        :param network: network code
        :type network: str
        :param station: station code
        :type station: str
        :return: (UTCDateTime, UTCDateTime)
        :rtype: tuple
        """
        dirlist = glob.glob(os.path.join(self.sds_root, '*', network, station))
        dirlist2 = [
            os.path.basename(
                os.path.dirname(os.path.dirname(i))) for i in dirlist]
        # We only want the folders with years
        l0 = fnmatch.filter(dirlist2, '1???')
        l1 = fnmatch.filter(dirlist2, '2???')
        del dirlist2
        dirlist = l0 + l1
        if not dirlist:
            warnings.warn(
                'Station %s.%s not in database' % (network, station),
                UserWarning)
            return (None, None)
        dirlist.sort()  # sort by time

        try:
            starttime = get_day_in_folder(
                self.sds_root, dirlist, network, station, '*', 'start')
            endtime = get_day_in_folder(
                self.sds_root, dirlist, network, station, '*', 'end')
        except FileNotFoundError as e:
            warnings.warn(e, UserWarning)
            return (None, None)
        return (starttime, endtime)

    def _load_remote(
        self, network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime, endtime: UTCDateTime,
            attach_response: bool) -> Stream:
        """
        Load data from remote resouce
        """
        print("Loading remotely.")
        try:
            st = self.rclient.get_waveforms(
                network, station, location, channel, starttime, endtime)
        except FDSNNoDataException as e:
            print(e)
            return Stream()
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                st.attach_response(self.inventory)
            except Warning:
                print("Updating inventory.")
                ninv = self.rclient.get_stations(
                    network=network, station=station, channel=channel,
                    starttime=starttime, endtime=endtime, level='response')
                if attach_response:
                    st.attach_response(ninv)
                self._write_inventory(ninv)
        if ((len(st) > 0) and not self.read_only):
            self._write_local_data(st)
        return st

    def _load_local(
        self, network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime, endtime: UTCDateTime, attach_response: bool,
            _check_times: bool) -> Stream or None:
        """
        Read data from local SDS structure.
        """
        # print("Loading locally ... ", end='')
        st = self.lclient.get_waveforms(
            network, station, location, channel, starttime, endtime)
        # Making sure that the order is correct for the next bit to work
        st.sort(['starttime'])
        if _check_times and (
            len(st) == 0 or (
                starttime < (st[0].stats.starttime-st[0].stats.delta) or (
                    endtime > st[-1].stats.endtime+st[-1].stats.delta))):
            # print("Failed")
            return None
        if attach_response:
            try:
                st.attach_response(self.inventory)
            except Exception:
                pass
        # print("Success")
        return st

    def read_inventory(self) -> Inventory:
        """
        Reads an up-to-date version of the corresponding Station Inventory.

        :return: Station Inventory
        :rtype: Inventory
        """
        self.inventory = read_inventory(os.path.join(self.inv_dir, '*.xml'))
        return self.inventory

    def select_inventory_or_load_remote(
            self, network: str, station: str) -> Inventory:
        """
        Checks whether the response for the provided station is in
        self.inventory if not the response will be fetched from remote.

        :param network: Network code
        :type network: str
        :param station: Station Code
        :type station: str
        :return: An Obspy Inventory object holding the response for the
            requested station.
        :rtype: Inventory
        """
        inv = self.inventory.select(network=network, station=station)
        if not len(inv):
            print('Station response not found ... loading from remote.')
            inv = self.rclient.get_stations(
                network=network, station=station,
                channel='*', level='response')
            self._write_inventory(inv)
        return inv

    def _write_local_data(self, st):
        """
        Write stream to local SDS structure.
        """
        # write the data
        for tr in st:
            starttime = tr.stats.starttime
            while starttime < tr.stats.endtime:
                endtime = UTCDateTime(
                    starttime.year, starttime.month, starttime.day) + 86400*1.5
                endtime = UTCDateTime(endtime.year, endtime.month, endtime.day)
                ttr = tr.copy().trim(starttime, endtime, nearest_sample=False)
                self._sds_write(ttr, starttime.year, starttime.julday)
                starttime = endtime

    def _sds_write(self, tr, year, julday):
        """
        Write max 1 day long trace to SDS structure.
        """
        filename = SDS_FMTSTR.format(
            network=tr.stats.network, station=tr.stats.station,
            location=tr.stats.location, channel=tr.stats.channel,
            year=year, doy=julday, sds_type=self.sds_type)
        filename = os.path.join(self.sds_root, filename)
        if os.path.exists(filename):
            rst = read(filename)
            rst.append(tr)
            rst.merge()
        else:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            rst = tr
        rst.write(filename, format='MSEED')

    def _write_inventory(self, ninv: Inventory):
        """write the inventory information"""
        # Save inventory by station name, like this no old information
        # can be deleted
        if self.inventory is not None:
            self.inventory += ninv
        else:
            self.inventory = ninv
        fname = os.path.join(
            self.inv_dir, f'{ninv[0].code}.{ninv[0][0].code}.xml')
        self.inventory.write(fname, format="STATIONXML", validate=True)

    def _generate_time_windows(
        self, network: str, station: str, channel: str, starttime: UTCDateTime,
            endtime: UTCDateTime, increment: int = 86400) -> Iterator[Stream]:
        """
        Generates time windows with the requested increment from the requested
        station.

        :param network: network code
        :type network: str
        :param station: station code
        :type station: str
        :param channel: Channel Code
        :type channel: str
        :param starttime: Starttime of the first window
        :type starttime: UTCDateTime
        :param endtime: Endtime of the last window
        :type endtime: UTCDateTime
        :param increment: increment and window length between the windows,
            defaults to 86400
        :type increment: int, optional
        :yield: Stream with every window
        :rtype: Iterator[Stream]
        """
        starttimes = starttime.timestamp + np.arange(
            0, endtime.timestamp-starttime.timestamp, increment)
        for start in starttimes:
            end = UTCDateTime(start + increment)
            st = self._load_local(
                network, station, '*', channel, UTCDateTime(start), end,
                attach_response=True, _check_times=False)
            yield st

    def compute_spectrogram(
        self, network: str, station: str, channel: str, starttime: UTCDateTime,
        endtime: UTCDateTime, win_len: int,
            read_increment: int = 86400) -> Tuple[
                np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes a time series of spectrograms for the requested station and
        channel.

        # Enter plotting function here later
        .. seealso:: Use function :func:`~seismic.plot` to generate

        :param network: network code
        :type network: str
        :param station: station code
        :type station: str
        :param channel: channel code
        :type channel: str
        :param starttime: starttime of the first window
        :type starttime: UTCDateTime
        :param endtime: endtime of the last window
        :type endtime: UTCDateTime
        :param win_len: Length of each time window in seconds
        :type win_len: int
        :param read_increment: Increment to read data in - does not
            influence the final result but has to be >= win_len,
            defaults to 86400
        :type read_increment: int, optional
        :return: Frequency vector, time vector and spectrogram matrix
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        if read_increment < win_len:
            raise ValueError(
                'read_increment must be >= win_len, got {} and {}'.format(
                    read_increment, win_len))
        data_gen = self._generate_time_windows(
            network, station, channel, starttime, endtime, read_increment)
        return spct_series_welch(data_gen, win_len)


class FS_Client(object):
    """
    Request Client for reading MSEED files from file system
    """
    def __init__(self, fs='SDS_archive'):
        """
        Initialize the client

        Refer to :class:`read_from_filesystem` for documentation
        """
        self.fs = fs

    def get_waveforms(
        self, network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime, endtime: UTCDateTime, trim: bool = True,
            debug: bool = False, **kwargs) -> Stream:
        """
        Read data from the local file system.
        """
        ID = network+'.'+station+'.'+location+'.'+channel
        if isinstance(starttime, UTCDateTime):
            starttime = starttime.datetime
        if isinstance(endtime, UTCDateTime):
            endtime = endtime.datetime
        st = read_from_filesystem(
            ID, starttime, endtime, self.fs, trim=trim, debug=debug)
        return st


IDformat = ['%NET', '%net', '%STA', '%sta', '%LOC', '%loc', '%CHA', '%cha']


def read_from_filesystem(
    ID: str, starttime: datetime.datetime, endtime: datetime.datetime,
    fs: str = 'SDS_archive', trim: bool = True,
        debug: bool = False) -> Stream:
    """
    Read data from a filesystem

    Function to read miniSEED data from a given file structure by specifying
    ID and time interval plus a description of the file system structure.

    :param ID: seedID of the channel to read (NET.STA.LOC.CHA)
    :type ID: string
    :param starttime: start time
    :type starttime: datetime.datetime
    :param endtime: end time
    :type endtime: datetime.datetime
    :param fs: file structure descriptor
    :type fs: list
    :param trim: switch for trimming of the stream
    :type trim: bool
    :param debug: print debugging information
    :type debug: bool

    :rtype: :class:`obspy.Stream`
    :return: data stream of requested data

    If the switch for trimming is False the whole stream in the files is
    returned.

    **File structure descriptor**
    fs is a list of strings or other lists indicating the elements in the
    file structure. Each item of this list is translated in one level of the
    directory structure. The first element is a string indicating the
    base_directory. The following elements can be strings to indicate
    one of the following:

    - %X as defined by datetime.strftime indicating an element of t
        the time. e.g. %H
    - %NET: network name or %net for lower case network name
    - %STA: station name or %sta for lower case
    - %CHA: channel name or %cha for lower case
    - %LOC: location or %loc for lower case location code
    - string with out %

    The format strings are replaced either with an element of the starttime
    if they correspond to a datetime specifyer or with the respective part
    of the seedID. A string withouta % sign is not changed. If more than one
    format string is required in one directory level the need to be
    separated within a sublist.

    A format string for the ID can be followed by a pair of braces including
    two strings that will be used to replace the first string with the
    second. This can be used if the filename contains part of the ID in a
    different form.

    If fs is a single string it is interpreted as the base directory
    'SDSdir' of a SeisComP Data Structure (SDS) with TYPE fixed to D
    `<SDSdir>/Year/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.DAY`
    This usage should be equivalent to `obspy.clients.filesystem.sds` client.


    :Example:

        Example for a station 'GJE' in network 'HEJSL' with channel 'BHZ' and
        location '00' with the start time 2010-12-24_11:36:30 and
        ``fs = ['base_dir','%Y','%b','%NET,['%j','_','%STA'','_T_',\
            "%CHA('BH','')", '.mseed']]``
        will be translated in a linux filename
        ``base_dir/2010/Nov/HEJSL/317_GJE_T_Z.mseed``

    .. note::

        If the data contain traces of different channels in the same file with
        different start and endtimes the routine will not work properly when a
        period spans multiple files.

    """

    # check input
    assert type(starttime) is datetime.datetime, \
        'starttime is not a datetime.datetime object: %s is type %s' % \
        (starttime, type(starttime))
    assert type(endtime) is datetime.datetime, \
        'endtime is not a datetime.datetime object: %s is type %s' % \
        (endtime, type(endtime))
    # check if fs is a string and set fs to SDS structure
    if isinstance(fs, str):
        fs = [fs, '%Y', '%NET', '%STA', ['%CHA', '.D'], [
            '%NET', '.', '%STA', '.', '%LOC', '.', '%CHA', '.D.', '%Y', '.',
            '%j']]
    # translate file structure string
    fpattern = _current_filepattern(ID, starttime, fs)
    if debug:
        print('Searching for files matching: %s\n at time %s\n' %
              (fpattern, starttime))
    st = _read_filepattern(fpattern, starttime, endtime, trim, debug)

    # if trace starts too late have a look in the previous section
    if (len(st) == 0) or (
            (st[0].stats.starttime-st[0].stats.delta).datetime > starttime):
        fpattern, _ = _adjacent_filepattern(ID, starttime, fs, -1)
        if debug:
            print('Searching for files matching: %s\n at time %s\n' %
                  (fpattern, starttime))
        st += _read_filepattern(fpattern, starttime, endtime, trim, debug)
        st.merge()
    thistime = starttime
    while ((len(st) == 0) or (st[0].stats.endtime.datetime < endtime)) & (
            thistime < endtime):
        fpattern, thistime = _adjacent_filepattern(ID, thistime, fs, 1)
        if debug:
            print('Searching for files matching: %s\n at time %s\n' %
                  (fpattern, thistime))
        if thistime == starttime:
            break
        st += _read_filepattern(fpattern, starttime, endtime, trim, debug)
        st.merge()
    if trim:
        st.trim(starttime=UTCDateTime(starttime), endtime=UTCDateTime(endtime))
    if debug:
        print('Following IDs are in the stream: ')
        for tr in st:
            print(tr.id)
        print('Selecting %s' % ID)
    st = st.select(id=ID)
    return st


def _read_filepattern(
    fpattern: str, starttime: datetime.datetime, endtime: datetime.datetime,
        trim: bool, debug: bool) -> Stream:
    """Read a stream from files whose names match a given pattern.
    """
    flist = glob.glob(fpattern)
    starttimes = []
    endtimes = []
    # first only read the header information
    for fname in flist:
        st = read(fname, headonly=True)
        starttimes.append(st[0].stats.starttime.datetime)
        endtimes.append(st[-1].stats.endtime.datetime)
    # now read the stream from the files that contain the period
    if debug:
        print('Matching files:\n')
        for (f, start, end) in zip(flist, starttimes, endtimes):
            print('%s from %s to %s\n' % (f, start, end))
    st = Stream()
    for ind, fname in enumerate(flist):
        if (starttimes[ind] < endtime) and (endtimes[ind] > starttime):
            if trim:
                st += read(
                    fname, starttime=UTCDateTime(starttime),
                    endtime=UTCDateTime(endtime))
            else:
                st += read(fname)
    try:
        st.merge()
    except Exception:
        print("Error merging traces for requested period!")
        st = Stream()
    return st


def _adjacent_filepattern(
    ID: str, starttime: datetime.datetime, fs: str,
        inc: int) -> Tuple[str, datetime.datetime]:
    """Return the file name that contains the data sequence prior to the one
    that contains the given time for a given file structure and ID.

    :param inc: either 1 for following of -1 for previous period
    :type inc: int
    """
    assert ((inc == 1) or (inc == -1)), " inc must either be 1 or -1"
    fname = ''
    flag = 0
    # find earlier time by turning back the time by one increment of the
    # last time indicator in the fs
    for part in fs[-1::-1]:
        if not isinstance(part, list):
            part = [part]
        for tpart in part[-1::-1]:
            if (not ((('(' in tpart) and (')' in tpart))
                or (tpart in IDformat))
                    and ('%' in tpart) and (flag == 0)):
                flag = 1
                if tpart in ['%H', '%I']:
                    thistime = starttime + inc * datetime.timedelta(hours=1)
                elif tpart == '%p':
                    thistime = starttime + inc * datetime.timedelta(hours=12)
                elif tpart in ['%a', '%A', '%w', '%d', '%j', '%-j']:
                    thistime = starttime + inc * datetime.timedelta(days=1)
                elif tpart in ['%U', '%W']:
                    thistime = starttime + inc * datetime.timedelta(days=7)
                elif tpart in ['%b', '%B', '%m']:
                    if starttime.month + inc == 0:
                        thistime = datetime.datetime(starttime.year-1,
                                                     12,
                                                     starttime.day,
                                                     starttime.hour,
                                                     starttime.minute,
                                                     starttime.second,
                                                     starttime.microsecond)
                    elif starttime.month + inc == 13:
                        thistime = datetime.datetime(starttime.year+1,
                                                     1,
                                                     starttime.day,
                                                     starttime.hour,
                                                     starttime.minute,
                                                     starttime.second,
                                                     starttime.microsecond)
                    else:
                        thistime = datetime.datetime(starttime.year,
                                                     starttime.month + inc,
                                                     starttime.day,
                                                     starttime.hour,
                                                     starttime.minute,
                                                     starttime.second,
                                                     starttime.microsecond)
                elif tpart in ['%y', '%Y']:
                    thistime = datetime.datetime(
                        starttime.year - inc, starttime.month, starttime.day,
                        starttime.hour, starttime.minute, starttime.second,
                        starttime.microsecond)
    fname = _current_filepattern(ID, thistime, fs)
    return fname, thistime


def _current_filepattern(
        ID: str, starttime: datetime.datetime, fs: str) -> str:
    """Return the file name that contains the data sequence that contains
    the given time for a given file structure and ID.
    """
    fname = ''
    for part in fs:
        if not isinstance(part, list):
            part = [part]
        fpartname = ''
        for tpart in part:
            fpartname += _fs_translate(tpart, ID, starttime)
        fname = os.path.join(fname, fpartname)
    return fname


def _fs_translate(part: str, ID: str, starttime: datetime.datetime) -> str:
    """Translate part of the file structure descriptor.
    """
    IDlist = ID.split('.')
    if ('(' in part) and (')' in part):
        trans = re.search('(.*?)', part).group(0)
    else:
        trans = None
    # in case there is something to translate remove from the filepart
    if trans:
        part = part.replace(trans, '')
    # if there is no %-sign it is a fixed string
    if '%' not in part:
        res = part
    # in case it belongs to the ID replace it with the respective ID-part
    if part in IDformat:
        idx = IDformat.index(part)
        res = IDlist[int(idx/2)]
        if idx % 2 != 0:  # idx is odd and IDformat is lowercase
            res = res.lower()
    # otherwise it must be part of the date string
    else:
        res = starttime.strftime(part)
    # replace if nesseccary
    if trans:
        transl = trans[1:-1].split(',')
        assert len(transl) == 2, "%s is not valid for replacement" % trans
        res = res.replace(
            transl[0].replace("'", ""), transl[1].replace("'", ""))
    return res


def get_day_in_folder(
    root: str, dirlist: List[str], network: str, station: str,
        channel: str, type: str) -> UTCDateTime:
    """
    Assuming that mseed files are by day (i.e., one file per day), this
    function will return the earliest or the latest day available (depending
    upon the argument passed as ``type``).

    :param root: The path to the sds root folder
    :type root: str
    :param dirlist: alphabetically sorted list of available years
    :type dirlist: List[str]
    :param network: The queried network code
    :type network: str
    :param station: The queried station code
    :type station: str
    :param channel: The queried channel code (wildcards allowed)
    :type channel: str
    :param type: either ``start`` or ``end``
    :type type: str
    :raises NotImplementedError: Unknown argument for type
    :raises FileNotFoundError: No files in db
    :return: The earliest starttime or latest endtime in utc
    :rtype: UTCDateTime
    """
    if type == 'start':
        i0 = 0
        ii = 1
    elif type == 'end':
        i0 = -1
        ii = -1
    else:
        raise NotImplementedError('Type has to be either start or end.')
    julday = None
    while not julday:
        # if the folder is empty julday will stay False
        year = dirlist[i0]
        julday = [i.split('.')[-1] for i in glob.glob(
            os.path.join(root, year, network, station, channel, '*'))]
        i0 += ii
    if not julday:
        raise FileNotFoundError(
            'Station %s.%s not in database' % (network, station))
    # make sure that 200 comes after 3, but before 300
    julday.sort(key=lambda li: (len(li), li))
    if type == 'start':
        return UTCDateTime(year=int(year), julday=julday[0])
    if type == 'end':
        # +24*3600 because else it's the starttime of latest day
        return UTCDateTime(year=int(year), julday=int(julday[-1])) + 24*3600
