'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 6th July 2021 09:18:14 am
Last Modified: Thursday, 21st October 2021 02:53:41 pm
'''

import os
import unittest
from unittest.mock import patch

import numpy as np
from obspy import UTCDateTime

from seismic.monitor import monitor
from seismic.monitor.dv import DV


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
    def setUp(self):
        self.p = os.path.curdir + os.path.sep
        return super().setUp()

    @patch('seismic.monitor.monitor.glob')
    def test_no_match_str(self, glob_mock):
        glob_mock.return_value = [
            '%sb-b.a-a.h5' % self.p, '%sa-a.b-b.h5' % self.p]
        net = {
            'network': 'AA',
            'station': 'BB'
        }
        for ii in monitor.corr_find_filter('.', net):
            self.assertFalse(len(ii))

    @patch('seismic.monitor.monitor.glob')
    def test_no_match_list0(self, glob_mock):
        glob_mock.return_value = [
            '%sb-b.a-a.h5' % self.p, '%sa-a.b-b.h5' % self.p]
        net = {
            'network': 'AA',
            'station': ['AA', 'BB']
        }
        for ii in monitor.corr_find_filter('.', net):
            self.assertFalse(len(ii))

    @patch('seismic.monitor.monitor.glob')
    def test_no_match_list1(self, glob_mock):
        glob_mock.return_value = [
            '%sb-b.a-a.h5' % self.p, '%sa-a.b-b.h5' % self.p]
        net = {
            'network': ['AA', 'BB'],
            'station': ['AA', 'BB']
        }
        for ii in monitor.corr_find_filter('.', net):
            self.assertFalse(len(ii))

    @patch('seismic.monitor.monitor.glob')
    def test_match_wildcard0(self, glob_mock):
        glob_mock.return_value = [
            '%sb-b.a-a.h5' % self.p, '%sa-a.b-b.h5' % self.p]
        net = {
            'network': '*',
            'station': '*'
        }
        n, s, i = monitor.corr_find_filter('.', net)

        self.assertListEqual(['b-b', 'a-a'], n)
        self.assertListEqual(['a-a', 'b-b'], s)
        self.assertListEqual(
            ['%sb-b.a-a.h5' % self.p, '%sa-a.b-b.h5' % self.p], i)

    @patch('seismic.monitor.monitor.glob')
    def test_match_wildcard1(self, glob_mock):
        glob_mock.return_value = [
            '%sb-b.a-a.h5' % self.p, '%sa-a.b-b.h5' % self.p]
        net = {
            'network': 'a',
            'station': '*'
        }
        n, s, i = monitor.corr_find_filter(os.path.curdir, net)
        self.assertListEqual(['a-a'], n)
        self.assertListEqual(['b-b'], s)
        self.assertListEqual(['%sa-a.b-b.h5' % self.p], i)

    @patch('seismic.monitor.monitor.glob')
    def test_match_list0(self, glob_mock):
        glob_mock.return_value = [
            '%sb-b.a-a.h5' % self.p, '%sa-a.b-b.h5' % self.p]
        net = {
            'network': ['a', 'b'],
            'station': ['a', 'b']
        }
        n, s, i = monitor.corr_find_filter(os.path.curdir, net)

        self.assertListEqual(['b-b', 'a-a'], n)
        self.assertListEqual(['a-a', 'b-b'], s)
        self.assertListEqual(
            ['%sb-b.a-a.h5' % self.p, '%sa-a.b-b.h5' % self.p], i)

    @patch('seismic.monitor.monitor.glob')
    def test_match_list1(self, glob_mock):
        glob_mock.return_value = [
            '%sb-b.a-a.h5' % self.p, '%sa-a.b-b.h5' % self.p]
        net = {
            'network': 'a',
            'station': ['a', 'b']
        }
        n, s, i = monitor.corr_find_filter(os.path.curdir, net)

        self.assertListEqual(['a-a'], n)
        self.assertListEqual(['b-b'], s)
        self.assertListEqual(['%sa-a.b-b.h5' % self.p], i)


class TestAverageComponents(unittest.TestCase):
    def test_differing_shape(self):
        sim0 = np.zeros((5, 5))
        sim1 = np.zeros((6, 6))
        corr0 = np.zeros((5))
        corr1 = np.zeros((6))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], {})
        dv1 = DV(corr1, corr1, ['stretch'], sim1, corr1, ['bla'], {})
        with self.assertRaises(ValueError):
            monitor.average_components([dv0, dv1])

    def test_differing_methods(self):
        sim0 = np.zeros((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], {})
        dv1 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['blub'], {})
        with self.assertRaises(TypeError):
            monitor.average_components([dv0, dv1])

    def test_contains_nans(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.nan*np.ones((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], {})
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], {})
        dv_av = monitor.average_components([dv0, dv1])
        self.assertTrue(np.all(dv0.sim_mat == dv_av.sim_mat))

    def test_result(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.random.random((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], {})
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], {})
        dv_av = monitor.average_components([dv0, dv1])
        self.assertTrue(np.allclose(
            np.mean([dv0.sim_mat, dv1.sim_mat], axis=0), dv_av.sim_mat))


if __name__ == "__main__":
    unittest.main()
