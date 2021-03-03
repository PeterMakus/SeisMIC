import os, datetime, glob

from obspy.clients.fdsn import Client as rClient
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.filesystem.sds import Client as lClient
from obspy import read_inventory, UTCDateTime, read, Stream
import warnings


SDS_FMTSTR = os.path.join(
    "{year}", "{network}", "{station}", "{channel}.{sds_type}",
    "{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}")

class Store_Client():
    """
    Client for requst and local storage of waveform data
    
    Request client that stores downloaded data for later local retrieval. 
    When reading stored data the client reads from the local copy of the data.
    Inventory data is stored in the folder `inventory` and attached to data
    that is read.
    """
    
    def __init__(self,Client,path,read_only=False):
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
        self.sds_root = path
        self.inv_name = os.path.join(path,"inventory","inventory.xml")
        if os.path.exists(self.inv_name):
            self.inventory = read_inventory(self.inv_name)
        else:
            self.inventory = read_inventory()
        if not os.path.exists(os.path.dirname(self.inv_name)):
            os.makedirs(os.path.dirname(self.inv_name))
        self.sds_type = "D"
        self.lclient = lClient(path, sds_type=self.sds_type, format="MSEED",
                               fileborder_seconds=self.fileborder_seconds,
                               fileborder_samples=self.fileborder_samples)
        self.rclient = Client
        self.read_only = read_only
        
        
    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime, attach_response=True):
        assert ~('?' in network+station+location+channel), "Only single channel requests supported."
        assert ~('*' in network+station+location+channel), "Only single channel requests supported."
        # try to load from local disk
        st = self._load_local(network, station, location, channel, starttime,
                              endtime, attach_response)
        if st is None:
            st = self._load_remote(network, station, location, channel, starttime,
                              endtime, attach_response)
        return st
    
    
    def _load_remote(self, network, station, location, channel, starttime,
                     endtime, attach_response):
        """
        Load data from remote resouce
        """
        print("Loading remotely.")
        try:
            st = self.rclient.get_waveforms(network, station, location, channel, starttime,
                                            endtime)
        except FDSNNoDataException as e:
            print(e)
            return Stream()
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                st.attach_response(self.inventory)
            except Warning as e:
                print("Updating inventory.")
                ninv = self.rclient.get_stations(network=network, station=station, channel=channel,
                                             starttime=starttime, endtime=endtime, level='response')
                if attach_response:
                    st.attach_response(ninv)
                self._write_inventory(ninv)
        if ((len(st) > 0) and not self.read_only):
            self._write_local_data(st)
        return st
    
    
    def _load_local(self, network, station, location, channel, starttime,
                    endtime, attach_response):
        """
        Read data from local SDS structure.
        """
        print("Loading locally ... ", end = '')
        st = self.lclient.get_waveforms(network, station, location, channel, starttime,
                           endtime)
        if ((len(st) == 0) or 
            ((starttime<(st[0].stats.starttime-st[0].stats.delta)) # potentially subtract 1 delta
             or (endtime>st[0].stats.endtime+st[0].stats.delta))):
            print("Failed")
            return None
        if attach_response:
            try:
                st.attach_response(self.inventory)
            except:
                pass
        print("Success")
        return st

    
    def _write_local_data(self, st):
        """
        Write stream to local SDS structure.
        """
        # write the data
        for tr in st:
            starttime = tr.stats.starttime
            while starttime < tr.stats.endtime:
                endtime = UTCDateTime(starttime.year,starttime.month,starttime.day)+86400*1.5
                endtime = UTCDateTime(endtime.year,endtime.month,endtime.day)
                ttr = tr.copy().trim(starttime,endtime,nearest_sample=False)
                self._sds_write(ttr,starttime.year,starttime.julday)
                starttime = endtime

                
    def _sds_write(self, tr, year, julday):
        """
        Write max 1 day long trace to SDS structure.
        """
        filename = SDS_FMTSTR.format(network=tr.stats.network, station=tr.stats.station,
                                      location=tr.stats.location, channel=tr.stats.channel,
                                      year=year, doy=julday, sds_type=self.sds_type)
        filename = os.path.join(self.sds_root,filename)
        if os.path.exists(filename):
            rst = read(filename)
            rst.append(tr)
        else:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            rst = tr
        rst.write(filename,format='MSEED')
        
                
    def _write_inventory(self,ninv):
        # write the inventory information
        if self.inventory is not None:
            self.inventory += ninv
        else:
            self.inventory = ninv
        self.inventory.write(self.inv_name,format="STATIONXML", validate=True)



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

    def get_waveforms(self, network, station, location, channel,
                      starttime, endtime, trim=True, debug=False, **kwargs):
        """
        Read data from the local file system.
        """
        ID = network+'.'+station+'.'+location+'.'+channel
        if isinstance(starttime,UTCDateTime):
            starttime = starttime.datetime
        if isinstance(endtime,UTCDateTime):
            endtime = endtime.datetime
        st = read_from_filesystem(ID, starttime, endtime, self.fs, trim=trim, debug=debug)
        return st
    


IDformat = ['%NET','%net','%STA','%sta','%LOC','%loc','%CHA','%cha']

def read_from_filesystem(ID,starttime,endtime,fs='SDS_archive',trim=True,debug=False):
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
    one of the following
     - %X as defined by datetime.strftime indicating an element of
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
    ''SDSdir' of a SeisComP Data Structure (SDS) with TYPE fixed to D
    <SDSdir>/Year/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.DAY
    This usage should be equvalent to obspy.clients.filesystem.sds client.


    :Exapmple:

    Example for a station 'GJE' in network 'HEJSL' with channel 'BHZ' and
    location '00' with the start time 2010-12-24_11:36:30 and \\
    ``fs = ['base_dir','%Y','%b','%NET,['%j','_','%STA'','_T_',"%CHA('BH','')",'.mseed']]``
    will be translated in a linux filename
    ``base_dir/2010/Nov/HEJSL/317_GJE_T_Z.mseed``

    :Note:

    If the data contain traces of different channels in the same file with
    different start and endtimes the routine will not work properly when the
    period spans multiple files.
    """

    #check input
    assert type(starttime) is datetime.datetime, \
        'starttime is not a datetime.datetime object: %s is type %s' % \
        (starttime, type(starttime))
    assert type(endtime) is datetime.datetime, \
        'endtime is not a datetime.datetime object: %s is type' % \
        (endtime, type(endtime))
    # check if fs is a string and set fs to SDS structure
    if isinstance(fs,str):
        fs  = [fs,'%Y','%NET','%STA',['%CHA','.D'],['%NET','.','%STA','.','%LOC','.','%CHA','.D.','%Y','.','%j']]
    # translate file structure string
    fpattern = _current_filepattern(ID,starttime,fs)
    if debug:
        print('Searching for files matching: %s\n at time %s\n' %
              (fpattern, starttime))
    st = _read_filepattern(fpattern, starttime, endtime,trim,debug)

    # if trace starts too late have a look in the previous section
    if (len(st)==0) or ((st[0].stats.starttime-st[0].stats.delta).datetime > starttime):
        fpattern, _ = _adjacent_filepattern(ID,starttime,fs,-1)
        if debug:
            print('Searching for files matching: %s\n at time %s\n' %
                  (fpattern, starttime))
        st += _read_filepattern(fpattern, starttime, endtime,trim,debug)
        st.merge()
    thistime = starttime
    while ((len(st)==0) or (st[0].stats.endtime.datetime < endtime)) & (thistime < endtime):
        fpattern, thistime = _adjacent_filepattern(ID,thistime,fs,1)
        if debug:
            print('Searching for files matching: %s\n at time %s\n' %
                  (fpattern, thistime))
        if thistime == starttime:
            break
        st += _read_filepattern(fpattern, starttime, endtime, trim, debug)
        st.merge()
    if trim:
        st.trim(starttime=UTCDateTime(starttime),endtime=UTCDateTime(endtime))
    if debug:
        print('Following IDs are in the stream: ')
        for tr in st:
            print(tr.id)
        print('Selecting %s' % ID)
    st = st.select(id=ID)
    return st


def _read_filepattern(fpattern, starttime, endtime, trim, debug):
    """Read a stream from files whose names match a given pattern.
    """
    flist = glob.glob(fpattern)
    starttimes = []
    endtimes = []
    # first only read the header information
    for fname in flist:
        st = read(fname,headonly=True)
        starttimes.append(st[0].stats.starttime.datetime)
        endtimes.append(st[-1].stats.endtime.datetime)
    # now read the stream from the files that contain the period
    if debug:
        print('Matching files:\n')
        for (f,start,end) in zip(flist,starttimes,endtimes):
            print('%s from %s to %s\n' % (f,start,end))
    st = Stream()
    for ind,fname in enumerate(flist):
        if (starttimes[ind] < endtime) and (endtimes[ind] > starttime):
            if trim:
                st += read(fname,starttime=UTCDateTime(starttime),endtime=UTCDateTime(endtime))
            else:
                st += read(fname)
    try:
        st.merge()
    except:
        print("Error merging traces for requested period!")
        st = Stream()
    return st


def _adjacent_filepattern(ID,starttime,fs,inc):
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
        if not isinstance(part,list):
            part = [part]
        for tpart in part[-1::-1]:
            if (not ((('(' in tpart) and (')' in tpart)) or (tpart in IDformat))
                                        and ('%' in tpart)
                                        and (flag == 0)):
                flag = 1
                if tpart in ['%H','%I']:
                    thistime = starttime + inc * datetime.timedelta(hours=1)
                elif tpart == '%p':
                    thistime = starttime + inc * datetime.timedelta(hours=12)
                elif tpart in ['%a','%A','%w','%d','%j','%-j']:
                    thistime = starttime + inc * datetime.timedelta(days=1)
                elif tpart in ['%U','%W']:
                    thistime = starttime + inc * datetime.timedelta(days=7)
                elif tpart in ['%b','%B','%m']:
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
                elif tpart in ['%y','%Y']:
                    thistime = datetime.datetime(starttime.year - inc,
                                                     starttime.month,
                                                     starttime.day,
                                                     starttime.hour,
                                                     starttime.minute,
                                                     starttime.second,
                                                     starttime.microsecond)
    fname = _current_filepattern(ID,thistime,fs)
    return fname, thistime


def _current_filepattern(ID,starttime,fs):
    """Return the file name that contains the data sequence that contains
    the given time for a given file structure and ID.
    """
    fname = ''
    for part in fs:
        if not isinstance(part,list):
            part = [part]
        fpartname = ''
        for tpart in part:
            fpartname += _fs_translate(tpart,ID,starttime)
        fname = os.path.join(fname,fpartname)
    return fname


def _fs_translate(part,ID,starttime):
    """Translate part of the file structure descriptor.
    """
    IDlist = ID.split('.')
    if ('(' in part) and (')' in part):
        trans = re.search('\(.*?\)',part).group(0)
    else:
        trans = None
    # in case there is something to translate remove from the filepart
    if trans:
        part = part.replace(trans,'')
    # if there is no %-sign it is a fixed string
    if not '%' in part:
        res = part
    # in case it belongs to the ID replace it with the respective ID-part
    if part in IDformat:
        idx = IDformat.index(part)
        res = IDlist[int(idx/2)]
        if idx%2 != 0:  # idx is odd and IDformat is lowercase
            res = res.lower()
    # otherwise it must be part of the date string
    else:
        res = starttime.strftime(part)
    # replace if nesseccary
    if trans:
        transl = trans[1:-1].split(',')
        assert len(transl) == 2, "%s is not valid for replacement" % trans
        res = res.replace(transl[0].replace("'",""),transl[1].replace("'",""))
    return res