'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 1st June 2021 10:42:03 am
Last Modified: Tuesday, 1st June 2021 04:18:12 pm
'''
import unittest
from unittest.mock import patch, MagicMock

from obspy import read, UTCDateTime
import numpy as np
from h5py._hl.group import Group

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


class TestAllTracesRecursive(unittest.TestCase):
    def setUp(self):
        # This is gonna be ugly
        #Doesn't work this way
        tag = '/test'
        net = ['TOTALLY']*3 + ['RANDOM']*3
        stat = ['RANDOM', 'BUT', 'SA', 'ME', 'LEN', 'GTH']
        channels = ['HHZ', 'HHE', 'HHN']
        starts = np.arange(0, 3600, 100)
        # List of paths
        self.paths = []
        # Create group mock
        self.g = MagicMock(spec=Group)
        self.g[tag] = MagicMock(spec=Group)
        self.g[tag].name = tag
        for n in net:
            self.g[tag][n] = MagicMock(spec=Group)
            self.g[tag][n].name = '/'.join([tag, n])
            for s in stat:
                self.g[tag][n][s] = MagicMock(spec=Group)
                self.g[tag][n][s].name = '/'.join([tag, n, s])
                for c in channels:
                    self.g[tag][n][s][c] = MagicMock(spec=Group)
                    self.g[tag][n][s][c].name = '/'.join([tag, n, s, c])
                    self.paths.extend(list(corr_hdf5.hierarchy.format(
                        tag='test', network=n, station=s, channel=c,
                        corr_st=UTCDateTime(st).format_fissures(),
                        corr_et=UTCDateTime(st+100).format_fissures())
                        for st in starts))
                    for st in starts:
                        t0 = UTCDateTime(st).format_fissures()
                        t1 = UTCDateTime(st+100).format_fissures()
                        self.g[tag][n][s][c][t0][t1].name = '/'.join(
                            [tag, n, c, t0, t1])

    def test_get_all(self):
        pattern = corr_hdf5.hierarchy.format(
            tag='test', network='TOTALLY', station='*', channel='*', corr_st='*',
            corr_et='*')
        pattern = pattern.replace('/*', '*')
        out = []
        out = corr_hdf5.all_traces_recursive(self.g, out, pattern)
        self.assertEqual(len(self.paths), len(out))


if __name__ == "__main__":
    unittest.main()
