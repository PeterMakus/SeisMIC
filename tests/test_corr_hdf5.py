'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 1st June 2021 10:42:03 am
Last Modified: Tuesday, 1st June 2021 12:08:03 pm
'''
import unittest
from unittest.mock import patch, MagicMock

from obspy import read, UTCDateTime
import numpy as np
import h5py

from miic3.db import corr_hdf5


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


# class TestAllTracesRecursive(unittest.TestCase):
#     def setUp(self):
#         tag = 'test'
#         net = ['TOTALLY']*3 + ['RANDOM']*3
#         stat = ['RANDOM', 'BUT', 'SA', 'ME', 'LEN', 'GTH']
#         channels = ['HHZ', 'HHE', 'HHN']
#         starts = np.arange(0, 3600)
        


if __name__ == "__main__":
    unittest.main()
