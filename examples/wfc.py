'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 17th July 2023 10:44:29 am
Last Modified: Monday, 17th July 2023 01:02:36 pm
'''

import os
# This tells numpy to only use one thread
# As we use MPI this is necessary to avoid overascribing threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from copy import deepcopy

import yaml
from obspy.clients.fdsn import Client
import numpy as np

from seismic.correlate.correlate import Correlator
from seismic.trace_data.waveform import Store_Client
from seismic.monitor.monitor import Monitor


# Path to the paramter file we created in the step before
params = 'params.yaml'

with open(params) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# You don't have to set this (could be None)
client = Client('GEOFON')
# root is the same as proj_dir in params.yaml
root = 'data'
sc = Store_Client(client, root)

# fbands to compute for
# freq_min = 0.06*1.5**np.arange(15)
# freq_min = [
#     0.06, 0.09, 0.135, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8]
# freq_max = [
#     0.12, 0.18, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 12.49]
freq_max = np.sort(np.round(12.49/1.5**np.arange(14), 2))
freq_min = np.round(freq_max/2, 2)

# Compute correlations
for fmin, fmax in zip(freq_min[:-1:2], freq_max[1::2]):
    copm = params['co']
    copm['subdir'] = f'corr_{fmin}_{fmax}'
    copm['corr_args']['TDpreProcessing'] = [
        {
            'function': 'seismic.correlate.preprocessing_td.detrend',
            'args': {'type': 'linear'}},
        {
            'function': 'seismic.correlate.preprocessing_td.TDfilter',
            'args': {
                'type': 'bandpass', 'freqmin': fmin, 'freqmax': fmax}}
    ]
    copm['corr_args']['FDpreProcessing'] = [
        {
            'function': 'seismic.correlate.preprocessing_fd.FDfilter',
            'args': {'flimit': [
                fmin/2,
                fmin, fmax,
                1.5*fmax if fmax < 8.25 else 12.49]}}]
    c = Correlator(sc, options=deepcopy(params))
    st = c.pxcorr()

# Compute waveform coherence
for ii in range(len(freq_min)//2):
    params['wfc']['freq_min'] = freq_min[2*ii:2*ii+2].tolist()
    params['wfc']['freq_max'] = freq_max[2*ii:2*ii+2].tolist()
    params['co']['subdir'] = f'corr_{freq_min[ii*2]}_{freq_max[2*ii+1]}'
    m = Monitor(deepcopy(params))
    m.compute_waveform_coherence_bulk()
