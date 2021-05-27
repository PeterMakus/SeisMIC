'''
UnitTests for the waveform module.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 15th March 2021 03:33:25 pm
Last Modified: Thursday, 27th May 2021 04:14:39 pm
'''

import unittest
from unittest import mock
import os

import numpy as np
from obspy import UTCDateTime

from miic3.trace_data import waveform


class TestStoreClient(unittest.TestCase):
    @mock.patch('miic3.trace_data.waveform.os.path.isdir')
    def setUp(self, isdir_mock):
        isdir_mock.return_value = True
        dir = os.path.join('/my', 'random', 'dir')
        self.outdir = os.path.join(dir, 'mseed')
        self.net = 'mynet'
        self.stat = 'mystat'
        self.sc = waveform.Store_Client('testclient', dir, read_only=True)

    def test_not_in_db(self):
        self.assertEqual(self.sc._get_times(self.net, self.stat), (None, None))

    @mock.patch('miic3.trace_data.waveform.glob.glob')
    def test_get_times(self, glob_mock):
        start = UTCDateTime(np.random.randint(0, 3e8))
        end = UTCDateTime(np.random.randint(4e8, 8e8))
        years = np.arange(start.year, end.year+1)
        # days in start year
        startdays = np.arange(start.julday, 366)
        # Days in end year
        enddays = np.arange(1, end.julday+1)
        yeardirs = [os.path.join(
            self.outdir, str(year), self.net, self.stat) for year in years]
        startdaydirs = [os.path.join(
            yeardirs[0], 'HHE.D', "%s.%s.%s.%s.%s.%s.%s" %
            (self.net, self.stat, '00', 'HHE', 'D', years[0], day))
            for day in startdays]

        enddaydirs = [os.path.join(
            yeardirs[-1], 'HHE.D',
            "%s.%s.%s.%s.%s.%s.%s" %
            (self.net, self.stat, '00', 'HHE', 'D', years[-1], day))
            for day in enddays]
        glob_mock.side_effect = [yeardirs, startdaydirs, enddaydirs]
        #
        # Check if the directory is correct
        self.assertEqual(self.sc.sds_root, self.outdir)
        timetup = self.sc._get_times(self.net, self.stat)
        #
        t_control = (
            UTCDateTime(year=start.year, julday=start.julday),
            UTCDateTime(year=end.year, julday=end.julday+1))
        self.assertEqual(timetup, t_control)

    @mock.patch('miic3.trace_data.waveform.glob.glob')
    def test_get_available_stations(self, glob_mock):
        net = ['TOTAL']*3 + ['RANDOM']*3
        stat = ['RANDOM', 'BUT', 'SA', 'ME', 'LEN', 'GTH']
        glob_mock.return_value = [
            os.path.join(self.outdir, '1980', nc, sc) for nc, sc in zip(
                net, stat)]
        control_ret = [[nc, sc] for nc, sc in zip(net, stat)]
        self.assertEqual(control_ret, self.sc.get_available_stations())

    def test_no_available_data(self):
        with self.assertRaises(FileNotFoundError):
            self.sc.get_available_stations()


if __name__ == "__main__":
    unittest.main()
