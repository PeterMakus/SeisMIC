import os
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
    Request client that stores downloaded data for later local retrieval
    """
    def __init__(self,Client,path,read_only=False):
        """
        Initialize the client
        
        :type CLient: obspy request client that supports the 'get_waveform' method
        :param Client: obspy request client that supports the 'get_waveform' method
        :type path: str
        :param path: path of the store's sds root directory
        :type read_only: Bool
        :param read_only:
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