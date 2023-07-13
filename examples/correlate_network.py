import os
# This tells numpy to only use one thread
# As we use MPI this is necessary to avoid overascribing threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from time import time

from obspy.clients.fdsn import Client
import yaml

from seismic.correlate.correlate import Correlator
from seismic.trace_data.waveform import Store_Client

# Path to the paramter file we created in the step before
params = 'params.yaml'
# adapt the parameters

with open(params, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params['net']['station'] = [
    'IR1', 'IR12', 'IR17', 'IR18', 'IR3', 'SV3', 'SV7', 'SV13']
params['co']['read_start'] = '2016-01-28 00:00:01.0'
params['co']['read_end'] = '2016-02-02 00:00:00.0'
params['co']['combination_method'] = 'betweenComponents'
# You don't have to set this (could be None)
client = Client('GEOFON')
# root is the same as proj_dir in params.yaml
root = 'data'
sc = Store_Client(client, root)

c = Correlator(sc, options=params)
print('Correlator initiated')
x = time()
st = c.pxcorr()
print('Correlation finished after', time()-x, 'seconds')
