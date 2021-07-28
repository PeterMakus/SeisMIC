'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 6th July 2021 09:18:14 am
Last Modified: Wednesday, 28th July 2021 10:35:18 am
'''

import unittest
from unittest.mock import patch

import numpy as np
from obspy import UTCDateTime

from seismic.monitor import monitor


class TestMakeTimeList(unittest.TestCase):
    def test_result(self):
        start_date = UTCDateTime(np.random.randint(0, 157788000))
        end_date = start_date + np.random.randint(31557600, 157788000)
        win_len = np.random.randint(1800, 2*86400)
        date_inc = np.random.randint(1800, 2*86400)
        st, et = monitor.make_time_list(
            start_date, end_date, date_inc, win_len)
        self.assertEqual(len(st), len(et))
        self.assertEqual(len(st), np.ceil((end_date-start_date)/date_inc))
        self.assertTrue(np.all(et-st == win_len))

    def test_end_before_start(self):
        start_date = UTCDateTime(np.random.randint(0, 157788000))
        end_date = start_date - np.random.randint(31557600, 157788000)
        win_len = np.random.randint(1800, 2*86400)
        date_inc = np.random.randint(1800, 2*86400)
        with self.assertRaises(ValueError):
            _ = monitor.make_time_list(start_date, end_date, date_inc, win_len)

    def test_neg_inc(self):
        start_date = UTCDateTime(np.random.randint(0, 157788000))
        end_date = start_date + np.random.randint(31557600, 157788000)
        win_len = np.random.randint(1800, 2*86400)
        date_inc = np.random.randint(-1800, 0)
        with self.assertRaises(ValueError):
            _ = monitor.make_time_list(start_date, end_date, date_inc, win_len)

    def test_neg_winlen(self):
        start_date = UTCDateTime(np.random.randint(0, 157788000))
        end_date = start_date + np.random.randint(31557600, 157788000)
        date_inc = np.random.randint(1800, 2*86400)
        win_len = np.random.randint(-1800, 0)
        with self.assertRaises(ValueError):
            _ = monitor.make_time_list(start_date, end_date, date_inc, win_len)


class TestCorrFindFilter(unittest.TestCase):
    @patch('seismic.monitor.monitor.glob')
    def test_no_match_str(self, glob_mock):
        glob_mock.return_value = [
            './b-b.a-a.h5', './a-a.b-b.h5']
        net = {
            'network': 'AA',
            'station': 'BB'
        }
        for ii in monitor.corr_find_filter('.', net):
            self.assertFalse(len(ii))

    @patch('seismic.monitor.monitor.glob')
    def test_no_match_list0(self, glob_mock):
        glob_mock.return_value = [
            './b-b.a-a.h5', './a-a.b-b.h5']
        net = {
            'network': 'AA',
            'station': ['AA', 'BB']
        }
        for ii in monitor.corr_find_filter('.', net):
            self.assertFalse(len(ii))

    @patch('seismic.monitor.monitor.glob')
    def test_no_match_list1(self, glob_mock):
        glob_mock.return_value = [
            './b-b.a-a.h5', './a-a.b-b.h5']
        net = {
            'network': ['AA', 'BB'],
            'station': ['AA', 'BB']
        }
        for ii in monitor.corr_find_filter('.', net):
            self.assertFalse(len(ii))

    @patch('seismic.monitor.monitor.glob')
    def test_match_wildcard0(self, glob_mock):
        glob_mock.return_value = [
            './b-b.a-a.h5', './a-a.b-b.h5']
        net = {
            'network': '*',
            'station': '*'
        }
        n, s, i = monitor.corr_find_filter('.', net)

        self.assertListEqual(['b-b', 'a-a'], n)
        self.assertListEqual(['a-a', 'b-b'], s)
        self.assertListEqual(['./b-b.a-a.h5', './a-a.b-b.h5'], i)

    @patch('seismic.monitor.monitor.glob')
    def test_match_wildcard1(self, glob_mock):
        glob_mock.return_value = [
            './b-b.a-a.h5', './a-a.b-b.h5']
        net = {
            'network': 'a',
            'station': '*'
        }
        n, s, i = monitor.corr_find_filter('.', net)
        self.assertListEqual(['a-a'], n)
        self.assertListEqual(['b-b'], s)
        self.assertListEqual(['./a-a.b-b.h5'], i)

    @patch('seismic.monitor.monitor.glob')
    def test_match_list0(self, glob_mock):
        glob_mock.return_value = [
            './b-b.a-a.h5', './a-a.b-b.h5']
        net = {
            'network': ['a', 'b'],
            'station': ['a', 'b']
        }
        n, s, i = monitor.corr_find_filter('.', net)

        self.assertListEqual(['b-b', 'a-a'], n)
        self.assertListEqual(['a-a', 'b-b'], s)
        self.assertListEqual(['./b-b.a-a.h5', './a-a.b-b.h5'], i)

    @patch('seismic.monitor.monitor.glob')
    def test_match_list1(self, glob_mock):
        glob_mock.return_value = [
            './b-b.a-a.h5', './a-a.b-b.h5']
        net = {
            'network': 'a',
            'station': ['a', 'b']
        }
        n, s, i = monitor.corr_find_filter('.', net)

        self.assertListEqual(['a-a'], n)
        self.assertListEqual(['b-b'], s)
        self.assertListEqual(['./a-a.b-b.h5'], i)


if __name__ == "__main__":
    unittest.main()
