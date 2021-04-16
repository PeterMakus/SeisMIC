'''
Manages the file format and class for correlations.

:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 16th April 2021 03:21:30 pm
Last Modified: Friday, 16th April 2021 04:02:15 pm
'''

import numpy as np
import h5py


class CorrelationDataBase(object):
    def __init__(self, path: str, mode: str = 'a'):
        # Create / read file
        if not path.split('.')[-1] == 'hdf5':
            path += '.hdf5'
        self.db_file = h5py.File(path, mode)

    def add_correlation(self, data: np.ndarray, header: dict):
        # There should be a CorrTrace and CorrStream object that are subclasses
        # of the obspy classes that can be used here.
        try :
                h5file.create_dataset("corr_data/"+t, data=data)
            except RuntimeError as e :
                print(("The appending dataset is corr_data/"+t+" in file "+filename))
                #sys.exit()
                raise e


def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in list(dic.items()):
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, float, int, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in list(h5file[path].items()):
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def corr_to_hdf5(data,stats,stats_tr1,stats_tr2,base_name,base_dir) :
    """ Output a correlation function to a hdf5 file.
    The hdf5 file contains three groups for the 3 stats dictionaries,
    and a "corr_data" group into which each correlation function
    is appended as a HDF5-dataset

    :type data: :class:`~numpy.ndarray`
    :param data: Correlation function to be written to hdf5 file
    :type stats: dictionary
    :param stats: Correlation stats determined by miic.core.corr_fun.combine_stats
    :type stats_tr1: dictionary
    :param stats_tr1: Trace stats for tr1
    :type stats_tr2: dictionary
    :param stats_tr2: Trace stats for tr2

    :type base_name: string
    :param base_name: Common "root" for every generated filename.
        It must not include underscores.
    :type base_dir: directory
    :param base_dir: Path where to save the files
    """

    _tr1dict = {'network': stats_tr1.network,
                'station': stats_tr1.station,
                'location': stats_tr1.location,
                'channel': stats_tr1.channel,
                'sampling_rate': stats_tr1.sampling_rate,
                'starttime': '%s' % stats_tr1.starttime,
                'endtime': '%s' % stats_tr1.endtime,
                'npts': int(stats_tr1.npts)}
    if 'sac' in stats_tr1:
        _tr1dict['stla'] = stats_tr1.sac.stla
        _tr1dict['stlo'] = stats_tr1.sac.stlo
        _tr1dict['stel'] = stats_tr1.sac.stel

    _tr2dict = {'network': stats_tr2.network,
                'station': stats_tr2.station,
                'location': stats_tr2.location,
                'channel': stats_tr2.channel,
                'sampling_rate': stats_tr2.sampling_rate,
                'starttime': '%s' % stats_tr2.starttime,
                'endtime': '%s' % stats_tr2.endtime,
                'npts': int(stats_tr2.npts)}
    if 'sac' in stats_tr2:
        _tr2dict['stla'] = stats_tr2.sac.stla
        _tr2dict['stlo'] = stats_tr2.sac.stlo
        _tr2dict['stel'] = stats_tr2.sac.stel

    _stats = {'network': stats.network,
              'station': stats.station,
              'location': stats.location,
              'channel': stats.channel,
              'sampling_rate': stats.sampling_rate,
              'starttime': '%s' % stats.starttime,
              'endtime': '%s' % stats.endtime,
              'npts': int(stats.npts)}
    if 'sac' in stats:
        _stats['stla'] = stats.sac.stla
        _stats['stlo'] = stats.sac.stlo
        _stats['stel'] = stats.sac.stel
        if np.all([x in stats.sac for x in ['evla', 'evlo', 'evel', 'az', 'baz', 'dist']]):
            _stats['evla'] = stats.sac.evla
            _stats['evlo'] = stats.sac.evlo
            _stats['evel'] = stats.sac.evel
            _stats['az'] = stats.sac.az
            _stats['baz'] = stats.sac.baz
            _stats['dist'] = stats.sac.dist

    # Determine file name and time
    corr_id=".".join([stats.network,stats.station,stats.location,stats.channel])
    filename = os.path.join(base_dir,base_name + '_' + corr_id.replace('-', '')+'.h5')
    t = max(_tr1dict['starttime'],_tr2dict['starttime'])
    time = '%s' % t
    time = time.replace('-', '').replace('.', '').replace(':', '')

    # If file doesn't exist create the stats groups and data in corr_data group
    if not os.path.exists(filename):
        create_path(base_dir)
        h5dicts={'stats_tr1':_tr1dict, 'stats_tr2':_tr2dict, 'stats':_stats,
                'corr_data':{t:data} }
        save_dict_to_hdf5(h5dicts, filename)
    # Else append data to corr_data group
    else :
        with h5py.File(filename, 'a') as h5file:
            try :
                h5file.create_dataset("corr_data/"+t, data=data)
            except RuntimeError as e :
                print(("The appending dataset is corr_data/"+t+" in file "+filename))
                #sys.exit()
                raise e

    return 0