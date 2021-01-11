##### from obspy import Trace, Stream, UTCDateTime, read
from obspy import Trace, Stream, UTCDateTime
from obspy.io.mseed import InternalMSEEDError
import numpy as np
from scipy import fft as sp_fft
from obspy.signal.util import next_pow_2
from copy import deepcopy
import os
import glob
import matplotlib.pyplot as plt


def plot(st, freq_keys=['?'], samp_keys=['?'], type_key='R', plot_type='lines', normalize=False, trafo=None):
    """Plot selected traces

    :tpye freq_keys: list
    :param freq_keys: list of frequency keys to plot
    :tpye samp_keys: list
    :param samp_keys: list of sampling rate keys to plot
    :type type_key: char
    :param type_key: one of 'R', 'M', 'S', 'C', 'D', 'E', 'F'
        'R': mean square amplitude derived from the distribution of small amplitude samples
        'M': mean amplitude of envelope
        'S': mean square amplitude of envelope
        'C': chi (mean log misfit of cumulative distribution function and sorted envelope samples)
        'D': chi_abs (mean of absolute value of log misfit of cumulative distribution function and sorted envelope samples)
        'E': chi restricted to the percentile range used to estimate 'R'
        'F': chi_abs restricted to the percentile range used to estimate 'R'
    :type plot_type: str
    :param plot_type: 'lines', 'color_panel'
    :type normalize: bool
    :param normaliz: normalize each trace individualy
    :type trafo: function or None
    :param trafo: function to apply to the data for transformation.
        It is expected to accept a numpy array as input and also return a numpy array with the transformed data,
    """
    # select traces to plot
    sst = Stream()
    for freq_key in freq_keys:
        for samp_key in samp_keys:
            channel = "?%s?" % type_key
            location = "%s%s" % (freq_key,samp_key)
            sst += st.select(location=location, channel=channel)
    # sort ferquencies
    locs = [tr.stats.location.upper() for tr in sst]
    inds = np.argsort(locs)
    fkchar = np.array([ord(locs[ind][0]) for ind in inds])
    # different frequencies present?
    if np.any(np.diff(fkchar)!=0):
        # any of these frequencies < 1Hz
        if np.any(fkchar>76):
            index = min(np.where(fkchar>76)[0])
            inds = np.roll(inds,-index)
    print([tr.stats.location for tr in sst])
    print(inds)
    if plot_type == 'lines':
        plt.figure(figsize=(8,4))
        for ind in inds:
            tr = sst[ind]
            times = [(tr.stats.starttime + dtim).datetime for dtim in (np.arange(tr.stats.npts)*tr.stats.delta)]
            if trafo:
                plt.plot(times, trafo(tr.data),'.')
            else:
                plt.plot(times, tr.data,'.')
    elif plot_type == 'color_panel':
        # check for different sampling rates
        deltas = np.array([tr.stats.delta for tr in sst])
        starttimes = [tr.stats.starttime for tr in sst]
        endtimes = [tr.stats.endtime for tr in sst]
        delta_min = np.min(deltas)
        starttime_min = min(starttimes)
        endtime_max = max(endtimes)
        # times for plotting in seconds starting at the earliest sample in the stream
        times = np.arange(int(np.ceil((endtime_max-starttime_min)/delta_min)))*delta_min
        pmat = np.zeros((len(sst),len(times)))
        for ind,tr in enumerate([sst[ind] for ind in inds]):
            #for ind,tr in enumerate(sst[inds]):
            ttimes = np.arange(tr.stats.npts)*tr.stats.delta + (tr.stats.starttime-starttime_min)
            pmat[ind,:] = np.interp(times,ttimes,tr.data)
        if normalize:
            pmat = pmat / np.nanmax(pmat,1,keepdims=True)
        if trafo:
            pmat = trafo(pmat)
        plt.imshow(pmat,aspect='auto',interpolation=None)
        plt.colorbar()
        return pmat






        
def frequency_dict(pow2_factor=0.5):
    """Return a dictionary defining a set of frequencies

    Define a set of logarithmically spaced frequnecies

    :type pow2_factor: float
    :param pow2_factor: the power of 2 increment in the 
        frequencies. Can be 0.5, 1, 1.5, 2 ... 
    """
    frequencies = {}
    # for Gaussian filters the freqs are (central freq, standard deviation)
    for ind in np.arange(0,5.6,pow2_factor):
        frequencies.update({chr(int(ind*2+65)):2.**ind})
    for ind in np.arange(-pow2_factor,-7.5,-pow2_factor):
        frequencies.update({chr(int(91+ind*2)):2.**(ind)})
    return frequencies

def sampling_dict():
    """Return a dictionary defining the sampling intervals
    """
    sampling = {'A':60, 'B':600, 'C':3600, 'D':3600*3,
                        'E':3600*12, 'F':3600*24, 'G':3600*24*3}
    return sampling


class Field_Amplitudes():
    """Estimate noise amplitudes from stream"""
    def __init__(self, frequencies=None, sampling=None, bandwidth_factor=0.125 , percentile_range=[1.,10], 
                 required_data=0.9, reftime=UTCDateTime(2000,1,1), padding=None, calc_chi=True):
        """
        frequencies: freq bands to be analysed e.g.{'one_letter_key':[central frequency, standard deviation of gaussian filter]}
        sampling: sampling intervals e.g. {'one letter key':sampling interval in seconds}
            the sampling interval corresponds to the length of the intervals in which the amplitude distribution in estimated
        percentile: the percentile at which the small amplitude Rayleigh statistics is estimated
        required_data: fraction of date to be present in interval to estimate statistics
        ref_time: reference time to adjust the locations of the samples
        calc_chi: calculate chi 
        bandwidth_factor: factor to be multiplied on the frequency to obtain the bandwidth
        """

        self.frequencies = frequencies or frequency_dict(pow2_factor=0.5)
        bws = 1./np.arange(1,11)
        assert bandwidth_factor in bws, \
            print('Select bandwidth_factor from %s' % [1./num for num in range(1,11)])
        self.bandwidth_factor = bandwidth_factor
        self.bandwidth_key = chr(int(91-1./bandwidth_factor))
        self.sampling = sampling or sampling_dict()
        self.tmax = np.max([self.sampling[key] for key in self.sampling.keys()])
        self.padding = padding or np.ceil(5.*1./np.min([frequencies[freq][0] for freq in frequencies]))
        self.percentile_range = percentile_range
        self.reftime = reftime
        self.required_data = required_data
        self.calc_chi = calc_chi
        
    def enter_trace(self, tr):
        assert isinstance(tr,Trace), "'tr' must be an obspy.trace object"
        assert tr.stats.endtime-tr.stats.starttime>2.*self.padding, "'tr' must be longer that twice the padding"
        self.trace = tr
        self._pre_process()
        # calc start and endtimes such that the timeing of samples does not depend 
        # on the starttime of the trace
        self.starttime = self.reftime + (np.round(((tr.stats.starttime+self.padding) -
                                                   self.reftime)/self.tmax) * self.tmax)
        #self.endtime = self.starttime + (np.ceil(((tr.stats.endtime-self.padding) -
        #                                          self.starttime)/self.tmax) * self.tmax)
        self.endtime = tr.stats.endtime-self.padding+1
        self.rst = None
        
    def _pre_process(self):
        """apply preprocessing"""
        if isinstance(self.trace.data,np.ma.MaskedArray):
            self.trace_mask = deepcopy(self.trace.data.mask)
            self.trace_mask = np.convolve(self.trace_mask.astype(float),
                np.ones(int(2.*0.8*self.padding*self.trace.stats.sampling_rate)),mode='same')
            self.trace_mask = self.trace_mask > 0.
        else:
            self.trace_mask = np.zeros(self.trace.stats.npts).astype(bool)
        self.trace = self.trace.split()
        self.trace.detrend()
        self.trace.taper(0.05,max_length=self.padding*0.8)
        # technical highpass
        self.trace.filter(type='highpass',freq=1./300)
        self.trace.taper(0.05,max_length=self.padding*0.8)
        self.trace.merge(fill_value=0.)
        # if trace was masked for gaps we will arrive here with a Stream object
        if isinstance(self.trace,Stream):
            self.trace = self.trace[0]

    def _hilbert_h(self,N):
        h = np.zeros(N)
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
        return h

    def _fft_truncate(self, N, Nout):
        """given a sequence of N fft coefficients (scipy.fftfreq) return those that are left when 
        the sequence is resampled to Nout samples"""
        if Nout % 2 == 1:
            No2 = (Nout-1) / 2
            outind = np.concatenate((np.arange(0,No2+1),np.arange(N-No2,N))).astype(int)
        else:
            No2 = Nout / 2
            outind = np.concatenate((np.arange(0,No2),np.arange(N-No2,N))).astype(int)
        return outind

        
        
    def calculate(self):
        #percentiles = np.logspace(-4,2,601)
        #percentile_range_inds = np.arange(np.argmin(np.abs(percentiles-self.percentile_range[0])),
        #                                       np.argmin(np.abs(percentiles-self.percentile_range[1])))
        # zero padding and Fourier transform
        N = next_pow_2(self.trace.stats.npts)
        current_sampling_rate = self.trace.stats.sampling_rate
        decimation = 1.
        # set order of calculations for sampling
        sorder = np.argsort([self.sampling[key] for key in sorted(self.sampling.keys())])
        # set the order of calculation for frequencies
        # start with highest freqency and gradually truncate the spectrum for downsampling
        forder = np.argsort([self.frequencies[key] for key in sorted(self.frequencies.keys())])[-1::-1]
        # the following is from
        # https://github.com/scipy/scipy/blob/v1.5.2/scipy/signal/signaltools.py#L2036-L2144
        spec = sp_fft.fft(self.trace.data, N, axis=0)
        freq = sp_fft.fftfreq(N,self.trace.stats.delta)
        h = self._hilbert_h(N)
        # object to accumulate results
        rst = Stream()
        # filtering and envelope calculation
        for find in forder:
            fkey = sorted(self.frequencies.keys())[find]
            center_freq = self.frequencies[fkey]
            bandwidth = self.frequencies[fkey] * self.bandwidth_factor
            if (center_freq + bandwidth) * 8 < current_sampling_rate:
                print("Decimate")
                decimation *= 2
                newN = int(N/2)
                spec_inds = self._fft_truncate(N, newN)
                current_sampling_rate = self.trace.stats.sampling_rate / decimation
                spec = spec[spec_inds]
                freq = freq[spec_inds]
                h = h[spec_inds]
                N = newN
            print(fkey, center_freq)
            ftaper = 1./(2.*np.pi*bandwidth**2) \
                * np.exp(-(np.abs(freq)-center_freq)**2/(2.*bandwidth**2))
            tspec = deepcopy(spec)*ftaper
            hilb = sp_fft.ifft(tspec * h) / decimation
            ttrace=Trace(header=self.trace.stats.copy())
            ttrace.stats.sampling_rate = current_sampling_rate
            ttrace.data = np.abs(hilb[:int(np.floor(self.trace.stats.npts / decimation))])
            ttrace.taper(0.05,max_length=self.padding*0.8)
            # apply mask
            ttrace.data[self.trace_mask[:int(len(ttrace.data)*decimation):int(decimation)]] = np.nan
            #chop of the segments that might have been affected by tapering
            ttrace.trim(starttime=ttrace.stats.starttime+self.padding,
                        endtime=ttrace.stats.endtime-self.padding)
            ttrace.trim(starttime=self.starttime, endtime=self.endtime,
                        pad=True, fill_value=np.nan)
            # calculation of the different sampling settings
            for sind in sorder:
                skey = sorted(self.sampling.keys())[sind]
                if self.sampling[skey]<100/center_freq:
                    continue
                print('s', skey, self.sampling[skey])
                # from a matrix with time in the 0-dimension and different segments
                # stacked in the 1-dimension
                ## number of samples per segments
                npts_seg = int(self.sampling[skey] * ttrace.stats.sampling_rate)
                ## number of samples in trace that a used
                npts = ttrace.stats.npts - ttrace.stats.npts%npts_seg
                ## number of segments
                n_seg = int(npts / npts_seg)
                ## sampling time is vertical (axis 0), segments stacked horizontal (axis 1)
                mat = np.reshape(ttrace.data[:npts],(n_seg,-1)).T
                # estimate mask for segments with too many nans
                nans = np.sum(np.isnan(mat),axis=0)/npts_seg
                nans = np.where(nans>(1.-self.required_data))[0]
                pri = [int(self.percentile_range[0]/100.*npts_seg),
                                       int(self.percentile_range[1]/100.*npts_seg)]
                smat = np.sort(mat,axis=0)
                tmp = np.sqrt(-2. * np.log(1 - np.arange(pri[0],pri[1])/npts_seg))
                tmp = smat[pri[0]:pri[1],:] / np.tile(tmp,(n_seg,1)).T
                sig_bar = np.exp(np.mean(np.log(tmp),axis=0))

                # calculate the mean squared amplitude of the Rayleigh distributed signal
                tMS = 2.*sig_bar**2
                tMS[nans] = np.nan
                # calculate the total mean squared amplitude
                MS = np.nanmean(mat**2,axis=0)
                MS[nans] = np.nan
                M = np.nanmean(mat,axis=0)
                M[nans] = np.nan
                L = np.nanmean(np.log10(mat),axis=0)*2.  # times two is for energy instead of amplitude
                L[nans] = np.nan
                # put the data in traces
                rtrace = Trace(header=ttrace.stats.copy())
                rtrace.stats.delta = self.sampling[skey]
                rtrace.stats.starttime = self.starttime + self.sampling[skey]/2.
                rtrace.stats.channel = "%sR%s" % (self.bandwidth_key,self.trace.stats.channel[2])
                rtrace.stats.location = "%s%s" % (fkey,skey)
                rtrace.data = tMS
                rst.append(rtrace.copy())
                rtrace.stats.channel = "%sS%s" % (self.bandwidth_key,self.trace.stats.channel[2])
                rtrace.data = MS
                rst.append(rtrace.copy())
                rtrace.stats.channel = "%sM%s" % (self.bandwidth_key,self.trace.stats.channel[2])
                rtrace.data = M
                rst.append(rtrace.copy())
                rtrace.stats.channel = "%sL%s" % (self.bandwidth_key,self.trace.stats.channel[2])
                rtrace.data = L
                rst.append(rtrace.copy())
                # calculate chi
                if self.calc_chi:
                    # Quantile
                    Q = np.atleast_2d(np.sqrt(-2.*np.log(1.-np.arange(1,npts_seg+1)/(npts_seg+1)))).T @ np.atleast_2d(sig_bar)
                    chi_mat = smat / Q
                    chi = np.nanmean(np.log(chi_mat),axis=0)
                    chi_abs = np.nanmean(np.abs(np.log(chi_mat)),axis=0)
                    chi_range = np.nanmean(np.log(chi_mat[pri[0]:pri[1],:]),axis=0)
                    chi_abs_range = np.nanmean(np.abs(np.log(chi_mat[pri[0]:pri[1],:])),axis=0)
                    rtrace.stats.channel = "%sC%s" % (self.bandwidth_key,self.trace.stats.channel[2])
                    rtrace.data = chi
                    rst.append(rtrace.copy())
                    rtrace.stats.channel = "%sD%s" % (self.bandwidth_key,self.trace.stats.channel[2])
                    rtrace.data = chi_abs
                    rst.append(rtrace.copy())
                    rtrace.stats.channel = "%sE%s" % (self.bandwidth_key,self.trace.stats.channel[2])
                    rtrace.data = chi_range
                    rst.append(rtrace.copy())
                    rtrace.stats.channel = "%sF%s" % (self.bandwidth_key,self.trace.stats.channel[2])
                    rtrace.data = chi_abs_range
                    rst.append(rtrace.copy())

        self.rst = rst
        return rst        
        
        
    
    def write(self, root='.', filename=None):
        filename = self.sds_filename()
        filename = filename or self.sds_filename()
        filename = os.path.join(root,filename)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.rst.write(os.path.join(root,filename),format="MSEED")
        
        
        
    def sds_filename(self):
        """Filename according to SeisComp3 SDS structure
        """
        SDS_FMTSTR = os.path.join("{year}", "{network}", "{station}", 
                                  "{sds_type}",
                                  "{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}.mseed")
        filename = SDS_FMTSTR.format(year=self.starttime.year,
                                     network=self.trace.stats.network,
                                     station=self.trace.stats.station,
                                     location="RA",
                                     channel=self.trace.stats.channel,
                                     sds_type="D",
                                     doy=self.starttime.julday)
        return filename

    
    
def production(ID, starttime, endtime, FA=None, client=None, outpath=None):
    FA = FA or Field_Amplitudes()
    client = client or Client('GFZ')
    outpath = outpath or '.'
    
    starttime = FA.reftime + (np.floor((starttime - FA.reftime)/FA.tmax) * FA.tmax)
    tinc = FA.tmax
    padding = FA.padding
        
    network, station, location, channel = ID.split('.')
    ttime = deepcopy(starttime)
    while ttime < endtime:
        print(ttime, "requesting data")
        try:
            rst = client.get_waveforms(network=network,station=station,location=location,channel=channel,
                                       starttime=ttime-padding-10,endtime=ttime+tinc+padding+10,attach_response=True)
        except InternalMSEEDError as e:
            print(e)
            ttime += tinc
            continue
        if not rst:
            ttime += tinc
            continue
        rst.remove_sensitivity()
        rst.merge(fill_value=None)
        print(rst, "Calculating")
        rst.trim(starttime=ttime-padding,endtime=ttime+tinc+padding+1,pad=True,fill_value=None)
        FA.enter_trace(rst[0])
        FA.calculate()
        FA.write(outpath)
        ttime += tinc
        
        
        
class Probabilistic_Field_Amplitudes():
    """Estimate probability densities of field amplitude distribution
    """
    def __init__(self, frequency, sampling, padding=None, percentiles=None,
                 amplitudes=None, scale_percentile_range=[1.,10]):
        """
        amplitudes: parameters of logarithmically spaces bins (min, max, number)
        """
        self.frequency = frequency
        self.sampling = sampling
        if padding is None:
            padding = 5./(self.frequency[0]-self.frequency[1])
        self.padding = padding
        if percentiles is None:
            percentiles = np.logspace(-4,2,300)
        self.percentiles = percentiles
        if amplitudes is None:
            amplitudes = [1e-1,1e2,200]
        self.amplitudes = amplitudes
        self.lamplitudes = [np.log10(self.amplitudes[0]),np.log10(self.amplitudes[1]),self.amplitudes[2]]
        self.scale_percentile_range = scale_percentile_range

        self.times = []
        self.probmat = np.zeros((int(self.amplitudes[2]),len(self.percentiles)))
        self.scaled_amplitude = []
        self.scale_percentile_inds = np.arange(np.argmin(np.abs(self.percentiles-scale_percentile_range[0])),
                                               np.argmin(np.abs(self.percentiles-scale_percentile_range[1])))

    def add_data(self, stream):
        """Add data to object
        """
        assert(isinstance(stream, Stream)), "'stream' must be an obspy Stream object: {}".format(type(stream))
        for trace in stream:
            print(trace.id)
            if hasattr(self,"ID"):
                assert(self.ID==trace.id), "IDs must match: {}, {}".format(self.ID,trace.id)
            else:
                self.ID = trace.id
            #self.times.append([trace.stats.starttime+self.padding,trace.stats.endtime-self.padding])
            self.trace = trace.copy()
            self._pre_process()
            # calculate amplitudevalues
            N = next_pow_2(self.trace.stats.npts)
            # the following is from
            # https://github.com/scipy/scipy/blob/v1.5.2/scipy/signal/signaltools.py#L2036-L2144
            spec = sp_fft.fft(self.trace.data, N, axis=0)
            freq = sp_fft.fftfreq(N,self.trace.stats.delta)
            h = np.zeros(N)
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
            # frequency domain filter
            ftaper = 1./(2.*np.pi*self.frequency[1]**2) \
                    * np.exp(-(np.abs(freq)-self.frequency[0])**2/(2.*self.frequency[1]**2))
            hilb = sp_fft.ifft(spec * ftaper * h)
            self.trace.data = np.abs(hilb[:self.trace.stats.npts])
            # put data in a matrix
            nsamp = int(self.sampling * self.trace.stats.sampling_rate)
            npts = self.trace.stats.npts - self.trace.stats.npts%nsamp
            mat = np.reshape(self.trace.data[:npts],(-1,nsamp))

            perc = np.nanpercentile(mat,self.percentiles,axis=1)
            for snum, seg in enumerate(np.arange(mat.shape[0])):
                #tperc = perc[:,seg]/((self.percentiles/100.)**0.5)
                tperc = np.sqrt(-perc[:,seg]**2/(2.*np.log(1.-self.percentiles/100.)))
                ascale = np.mean(tperc[self.scale_percentile_inds])
                #ascale = 1e-9
                self.scaled_amplitude.append(ascale)
                self.times.append((trace.stats.starttime + self.sampling*snum).datetime)
                tperc /= ascale
                ind = np.floor((np.log10(tperc)-self.lamplitudes[0])/(self.lamplitudes[1]-self.lamplitudes[0]) * self.lamplitudes[2]).astype(int)
                ind[ind<0] = 0
                ind[ind>=self.amplitudes[2]] = self.amplitudes[2]-1
                #print(ind, self.probmat.shape)
                self.probmat[ind,np.arange(len(self.percentiles),dtype=int)] += 1

        
        
        
        
        
    def _pre_process(self):
        """apply preprocessing"""
        self.trace.split()
        #self.trace.data = self.trace.data.filled(fill_value=0)
        self.trace.detrend()
        self.trace.taper(0.01,max_length=self.padding*0.8)
        # technical highpass
        self.trace.filter(type='highpass',freq=1./300)
        self.trace.taper(0.05,max_length=self.padding*0.8)
        try:
            self.trace.merge()
        except:
            pass


    def plot(self,normalize=True, show_Rayleigh=True):
        """Plot the probability distribution
        """
        fig = plt.figure(figsize=(6,6))
        a1 = fig.add_axes((0.15,0.4,0.7,0.55))
        a2 = fig.add_axes((0.15,0.15,0.7,0.15))
        if normalize:
            probmat = self.probmat/np.tile(np.max(self.probmat,axis=0),(self.probmat.shape[0],1))
        else:
            probmat = self.probmat
        a1.imshow(probmat,origin='lowerleft',aspect='auto',
                  extent=[np.log10(self.percentiles[0]),np.log10(self.percentiles[-1]),-1,2])
        if show_Rayleigh:
            #cumulative Rayleigh
            sig = 1./np.sqrt(2)
            amp = np.sqrt(-np.log(1-self.percentiles/100.))#* np.sqrt(2.*sig**2)
            a1.plot(np.log10(self.percentiles), np.log10(amp*(self.percentiles/100)**-0.5),'r-')
        a1.set_ylim(-0.5,1)
        a1.plot([np.log10(self.scale_percentile_range[0]),
                 np.log10(self.scale_percentile_range[1])],
                [a1.get_ylim()[0],a1.get_ylim()[0]],'w-',lw=6)
        xt = a1.get_xticks()
        xl = []
        for x in xt:
            xl.append("$10^{%d}$" % x)
        a1.set_xticklabels(xl)
        yt = np.arange(self.lamplitudes[0],self.lamplitudes[1]+0.1)
        yl = []
        for y in yt:
            yl.append("$10^{%d}$" % y)
        a1.set_yticks(yt)
        a1.set_yticklabels(yl)
        a1.set_ylim(-0.5,1)
        a1.set_xlabel('percentile')
        a1.set_ylabel('normalized scale parameter')
        a2.plot(self.times, self.scaled_amplitude,'-')
        a2.set_ylabel('scale parameter')

    def save(self,filename):
        np.savez(filename,
                 ID=self.ID,
                 probmat=self.probmat,
                 times=self.times,
                 scaled_amplitude=self.scaled_amplitude,
                 scale_percentile_range=self.scale_percentile_range,
                 percentiles=self.percentiles,
                 amplitudes=self.amplitudes,
                 frequency=self.frequency,
                 padding=self.padding,
                 sampling=self.sampling)
        

def load_Probabilistic_Field_Amplitudes(filename):
    dat = np.load(filename)
    pfa = Probabilistic_Field_Amplitudes(frequency=dat['frequency'],
                                         sampling=dat['sampling'],
                                         padding=dat['padding'],
                                         percentiles=dat['percentiles'],
                                         amplitudes=dat['amplitudes'],
                                         scale_percentile_range=dat['scale_percentile_range'])
    pfa.probmat=dat['probmat']
    pfa.times=dat['times']
    pfa.scaled_amplitude=dat['scaled_amplitude']
    pfa.ID=dat['ID']
    return pfa

def gauss_filter(st, frequency, width):
    """
    Filter stream witha a Gaussian filter
    
    :type st: obspy.stream.Stream
    :param st: stream to be filtered
    :type frequency: float
    :param frequency: central frequency of filter pass band in Hz
    :type width: float
    :param width: band width of pass band in Hz (standard deviation)
    """
    fst = st.copy()
    for tr in fst:
        N = next_pow_2(tr.stats.npts)
        spec = sp_fft.fft(tr.data, N, axis=0)
        freq = sp_fft.fftfreq(N,tr.stats.delta)
        ftaper = 1./(2.*np.pi*width**2) \
                    * np.exp(-(np.abs(freq)-frequency)**2/(2.*width**2))
        tr.data = (sp_fft.ifft(spec * ftaper).real)[:tr.stats.npts]
    return fst


def fit_Rayleigh_distribution(x,p=1.):
    """Fit the scale parameter of the Rayleigh distribution
    
    x: 1 or 2 dimensional array containing the seismic envelope data
        time axis is oriented along the first dimension (0).
        The array for three traces with 1000 samples has shape (1000,3).
    P: percentile of data points used to calculate the scale parameter.
        p should be small enough
    return: (sig, RayRMS, RMS) where 'sig' is the scale parameter of the Rayleigh
        distribution, 'RayMS' is the root mean square of the Rayleigh
        distributed values and 'RMS' is the root mean square of the
        data.
    """
    assert ((p<100) & (p>0)), "The percentile 'p' must be in the range 0<p<100: %f" % p
    xp = np.percentile(x,p,axis=0)
    #sig = p/(2.*q) + np.sqrt((p/(2.*q))**2-p**3/(2.*q))
    sig = np.sqrt(-xp**2./(2.*np.log(1.-p/100.)))
    RayMS = np.sqrt(2.)*sig
    RMS = np.sqrt(np.mean(x**2))
    return sig, RayRMS, RMS



def remove_non_Rayleigh_amp(env, percentile_range=[0.1, 10]):
    """
    Remove the non-Rayleigh amplitudes from an envelope
    """
    import pdb
    pdb.set_trace()
    renv = deepcopy(env)
    perc = np.logspace(np.log10(percentile_range[0]),
                       np.log10(percentile_range[1]),100)
    xp = np.percentile(renv,perc,axis=0)
    sig = np.mean(np.sqrt(-xp**2./(2.*np.log(1.-perc/100.))))
    amps_arg = np.argsort(renv)
    ray_amps = rayleigh.ppf(np.arange(len(env))/len(env),scale=sig)
    renv[amps_arg] = ray_amps
    return renv

