import os
# This tells numpy to only use one thread
# As we use MPI this is necessary to avoid overascribing threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import yaml

from seismic.monitor.monitor import Monitor

yaml_f = 'params.yaml'

with open(yaml_f, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params['net']['station'] = [
    'IR1', 'IR12', 'IR17', 'IR18', 'IR3',  'SV3', 'SV7', 'SV13']

params['dv']['start_date'] = '2016-01-28 00:00:00.0'
params['dv']['end_date'] = '2016-02-02 00:00:00.0'
params['dv']['tw_len'] = 20

m = Monitor(params)
m.compute_velocity_change_bulk()
