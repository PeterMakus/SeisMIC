'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 1st June 2021 10:42:03 am
Last Modified: Wednesday, 2nd June 2021 02:45:04 pm
'''
import unittest
from unittest.mock import patch, MagicMock

from obspy import read, UTCDateTime
import numpy as np
import h5py
from obspy.core.trace import Stats

from miic3.db import corr_hdf5
from miic3.correlate.stream import CorrStream


class TestConvertHeaderToHDF5(unittest.TestCase):
    def setUp(self):
        tr = read()[0]
        tr.decimate(4)  # so processing key is not empty
        self.stats = tr.stats

    def test_no_utc(self):
        # Check that all utcdatetime objects are now strings
        dataset = MagicMock()
        dataset.attrs = {}
        corr_hdf5.convert_header_to_hdf5(dataset, self.stats)
        for v in dataset.attrs.values():
            self.assertNotIsInstance(v, UTCDateTime)

    def test_length(self):
        # Check that all keys are transferred
        dataset = MagicMock()
        dataset.attrs = {}
        corr_hdf5.convert_header_to_hdf5(dataset, self.stats)
        self.assertEqual(dataset.attrs.keys(), self.stats.keys())


class TestReadHDF5Header(unittest.TestCase):
    def test_result(self):
        dataset = MagicMock()
        dataset.attrs = {}
        tr = read()[0]
        tr.decimate(4)  # to put something into processing
        stats = tr.stats
        corr_hdf5.convert_header_to_hdf5(dataset, stats)
        self.assertEqual(corr_hdf5.read_hdf5_header(dataset), stats)


def create_group_mock(d: dict, name: str, group: bool):
    """
    This is supposed to immitate the properties of
    :class:`h5py._hl.group.Group`

    :param d: dictionary
    :type d: dict
    :return: the mocked class
    :rtype: MagicMock
    """
    if group:
        m = MagicMock(spec=h5py._hl.group.Group)
    else:
        m = MagicMock()
    m.name = name
    m.__getitem__.side_effect = d.__getitem__
    m.__iter__.side_effect = d.__iter__
    m.__contains__.side_effect = d.__contains__
    m.values.side_effect = d.values
    return m

class TestAllTracesRecursive(unittest.TestCase):
    # The only thing I can do here is testing whether the conditions work
    @patch('miic3.db.corr_hdf5.read_hdf5_header')
    def test_is_np_array(self, read_header_mock):
        read_header_mock.return_value = None
        d = {'a': create_group_mock({}, 'testname', False),
        'b': create_group_mock({}, 'different_name', False)}
        #ndarray_mock.side_effect = d.keys()
        g = create_group_mock(d, 'outer_group', True)
        st = CorrStream()
        st = corr_hdf5.all_traces_recursive(g, st, 'testname')
        self.assertEqual(st.count(), 1)
        st = corr_hdf5.all_traces_recursive(g, st.clear(), 'different_name')
        self.assertEqual(st.count(), 1)
        st = corr_hdf5.all_traces_recursive(g, st.clear(), '*name')
        self.assertEqual(st.count(), 2)
        st = corr_hdf5.all_traces_recursive(g, st, 'no_match')
        self.assertEqual(st.count(), 0)

    @patch('miic3.db.corr_hdf5.read_hdf5_header')
    @patch('miic3.db.corr_hdf5.fnmatch.fnmatch')
    def test_recursive(self, read_header_mock, fnmatch_mock):
        # For this we need to patch fnmatch as well, as the names here aren't
        # full path
        read_header_mock.return_value = None
        fnmatch_mock.return_value = True
        d_inner = {'a': create_group_mock({}, 'testname', False),
        'b': create_group_mock({}, 'different_name', False)}
        d_outer = {'A': create_group_mock(d_inner, 'outer_group0', True),
                    'B': create_group_mock(d_inner, 'outer_group1', True)}
        g = create_group_mock(d_outer, 'outout', True)
        st = CorrStream()
        
        st = corr_hdf5.all_traces_recursive(g, st, 'doesnt_matter')
        self.assertEqual(st.count(), 4)

if __name__ == "__main__":
    unittest.main()
