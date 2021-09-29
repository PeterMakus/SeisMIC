'''
UnitTests for the waveform module.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 15th March 2021 03:33:25 pm
Last Modified: Wednesday, 29th September 2021 12:00:42 pm
'''

import unittest
from unittest import mock
import os
import warnings

import numpy as np
from obspy import UTCDateTime

from seismic.trace_data import waveform


class TestStoreClient(unittest.TestCase):
    @mock.patch('seismic.trace_data.waveform.os.listdir')
    @mock.patch('seismic.trace_data.waveform.os.path.isdir')
    def setUp(self, isdir_mock, listdir_mock):
        isdir_mock.return_value = True
        listdir_mock.return_value = False
        dir = os.path.join('%smy' % os.path.sep, 'random', 'dir')
        self.outdir = os.path.join(dir, 'mseed')
        self.net = 'mynet'
        self.stat = 'mystat'
        self.sc = waveform.Store_Client('testclient', dir, read_only=True)

    def test_not_in_db(self):
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(
                self.sc._get_times(self.net, self.stat), (None, None))
            self.assertEqual(len(w), 1)

    @mock.patch('seismic.trace_data.waveform.glob.glob')
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
            UTCDateTime(year=end.year, julday=end.julday)+24*3600)
        self.assertEqual(timetup, t_control)

    @mock.patch('seismic.trace_data.waveform.glob.glob')
    def test_get_times_year_change(self, glob_mock):
        # Just checking as there is no julday 366 in most years
        start = UTCDateTime(year=2015, month=12, day=31, hour=1)
        end = start + 24*3600
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
            UTCDateTime(year=end.year, julday=end.julday)+24*3600)
        self.assertEqual(timetup, t_control)

    @mock.patch('seismic.trace_data.waveform.glob.glob')
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
