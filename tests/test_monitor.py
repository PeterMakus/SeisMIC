'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 6th July 2021 09:18:14 am
Last Modified: Friday, 18th February 2022 03:08:44 pm
'''

import os
import unittest
from unittest import mock
from unittest.mock import patch
import warnings

import numpy as np
from obspy import UTCDateTime

from seismic.monitor import monitor
from seismic.monitor.dv import DV
from seismic.correlate.stats import CorrStats


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


class TestAverageDVbyCoords(unittest.TestCase):
    def test_none_within_filt(self):
        cstats = CorrStats()
        lat = (-10, 10)
        lon = (0, 10)
        cstats['stla'] = cstats['evla'] = lat[1] + np.random.randint(1, 60)
        cstats['stlo'] = cstats['evlo'] = lon[0] - np.random.randint(1, 89)
        cstats['stel'] = cstats['evel'] = 0
        dv = DV(
            np.zeros(5), np.zeros(5), 'bla', np.zeros((5, 5)), np.zeros(5),
            'dd', cstats)
        with self.assertRaises(ValueError):
            monitor.average_dvs_by_coords([dv], lat, lon)

    @mock.patch('seismic.monitor.monitor.average_components')
    def test_result(self, av_comp_mock: mock.MagicMock):
        cstats = CorrStats()
        lat = (-10, 10)
        lon = (0, 10)
        cstats['stla'] = cstats['evla'] = lat[1] + np.random.randint(1, 60)
        cstats['stlo'] = cstats['evlo'] = lon[0] + np.random.randint(0, 10)
        cstats['stel'] = cstats['evel'] = 0
        dv = DV(
            np.zeros(5), np.zeros(5), 'bla', np.zeros((5, 5)), np.zeros(5),
            'dd', cstats)
        cstats2 = CorrStats()
        cstats2['stla'] = cstats2['evla'] = lat[1] - np.random.randint(0, 20)
        cstats2['stlo'] = cstats2['evlo'] = lon[0] + np.random.randint(0, 10)
        cstats2['stel'] = cstats2['evel'] = 0
        dv2 = DV(
            np.zeros(5), np.zeros(5), 'bla', np.zeros((5, 5)), np.zeros(5),
            'dd', cstats2)
        av_comp_mock.return_value = (dv2, 'blub')
        av_dv, std = monitor.average_dvs_by_coords(
            [dv, dv2], lat, lon, return_std=True)
        av_comp_mock.assert_called_once_with([dv2], True)
        s = av_dv.stats
        np.testing.assert_array_equal([s.network, s.station], 'geoav')
        np.testing.assert_array_equal([s.stel, s.evel], 2*[(-1e6, 1e6)])
        np.testing.assert_array_equal([s.stlo, s.evlo], 2*[lon])
        np.testing.assert_array_equal([s.stla, s.evla], 2*[lat])
        self.assertEqual(std, 'blub')

    @mock.patch('seismic.monitor.monitor.average_components')
    def test_result2(self, av_comp_mock: mock.MagicMock):
        cstats = CorrStats()
        lat = (-10, 10)
        lon = (0, 10)
        cstats['stla'] = cstats['evla'] = lat[1] - np.random.randint(0, 20)
        cstats['stlo'] = cstats['evlo'] = lon[0] + np.random.randint(0, 10)
        cstats['stel'] = cstats['evel'] = 1000
        dv = DV(
            np.zeros(5), np.zeros(5), 'bla', np.zeros((5, 5)), np.zeros(5),
            'dd', cstats)
        cstats2 = CorrStats()
        cstats2['stla'] = cstats2['evla'] = lat[1] - np.random.randint(0, 20)
        cstats2['stlo'] = cstats2['evlo'] = lon[0] + np.random.randint(0, 10)
        cstats2['stel'] = cstats2['evel'] = 0
        dv2 = DV(
            np.zeros(5), np.zeros(5), 'bla', np.zeros((5, 5)), np.zeros(5),
            'dd', cstats2)
        av_comp_mock.return_value = dv2
        av_dv, std = monitor.average_dvs_by_coords(
            [dv, dv2], lat, lon, el=(-100, 100), return_std=False)
        av_comp_mock.assert_called_once_with([dv2], False)
        s = av_dv.stats
        np.testing.assert_array_equal([s.network, s.station], 'geoav')
        np.testing.assert_array_equal([s.stel, s.evel], 2*[(-100, 100)])
        np.testing.assert_array_equal([s.stlo, s.evlo], 2*[lon])
        np.testing.assert_array_equal([s.stla, s.evla], 2*[lat])
        self.assertIsNone(std)

    @mock.patch('seismic.monitor.monitor.average_components')
    def test_result_already_av(self, av_comp_mock: mock.MagicMock):
        cstats = CorrStats()
        lat = (-10, 10)
        lon = (0, 10)
        cstats['stla'] = cstats['evla'] = lat[1] - np.random.randint(0, 20)
        cstats['stlo'] = cstats['evlo'] = lon[0] + np.random.randint(0, 10)
        cstats['stel'] = cstats['evel'] = 1
        cstats['channel'] = 'av'
        dv = DV(
            np.zeros(5), np.zeros(5), 'bla', np.zeros((5, 5)), np.zeros(5),
            'dd', cstats)
        cstats2 = CorrStats()
        cstats2['stla'] = cstats2['evla'] = lat[1] - np.random.randint(0, 20)
        cstats2['stlo'] = cstats2['evlo'] = lon[0] + np.random.randint(0, 10)
        cstats2['stel'] = cstats2['evel'] = 0
        dv2 = DV(
            np.zeros(5), np.zeros(5), 'bla', np.zeros((5, 5)), np.zeros(5),
            'dd', cstats2)
        av_comp_mock.return_value = dv2
        with warnings.catch_warnings(record=True) as w:
            av_dv, std = monitor.average_dvs_by_coords(
                [dv, dv2], lat, lon, el=(-100, 100), return_std=False)
            self.assertEqual(len(w), 1)
        av_comp_mock.assert_called_once_with([dv2], False)
        s = av_dv.stats
        np.testing.assert_array_equal([s.network, s.station], 'geoav')
        np.testing.assert_array_equal([s.stel, s.evel], 2*[(-100, 100)])
        np.testing.assert_array_equal([s.stlo, s.evlo], 2*[lon])
        np.testing.assert_array_equal([s.stla, s.evla], 2*[lat])
        self.assertIsNone(std)


class TestAverageComponents(unittest.TestCase):
    def test_differing_shape(self):
        sim0 = np.zeros((5, 5))
        sim1 = np.zeros((6, 6))
        corr0 = np.zeros((5))
        corr1 = np.zeros((6))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr1, corr1, ['stretch'], sim1, corr1, ['bla'], CorrStats())
        with self.assertRaises(ValueError):
            monitor.average_components([dv0, dv1])

    def test_differing_methods(self):
        sim0 = np.zeros((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['blub'], CorrStats())
        with self.assertRaises(TypeError):
            monitor.average_components([dv0, dv1])

    def test_differing_2ndax(self):
        sim0 = np.zeros((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(
            corr0, corr0, ['stretch'], sim0, corr0+1, ['bla'], CorrStats())
        with self.assertRaises(ValueError):
            monitor.average_components([dv0, dv1])

    def test_contains_nans(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.nan*np.ones((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], CorrStats())
        dv_av = monitor.average_components([dv0, dv1])
        self.assertTrue(np.all(dv0.sim_mat == dv_av.sim_mat))

    def test_result(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.random.random((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], CorrStats())
        dv_av = monitor.average_components([dv0, dv1])
        self.assertTrue(np.allclose(
            np.mean([dv0.sim_mat, dv1.sim_mat], axis=0), dv_av.sim_mat))

    def test_result_already_av(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.random.random((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv0.stats['channel'] = 'av'
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], CorrStats())
        with warnings.catch_warnings(record=True) as w:
            dv_av = monitor.average_components([dv0, dv1])
            self.assertEqual(len(w), 1)
        np.testing.assert_array_equal(dv1.sim_mat, dv_av.sim_mat)

    def test_contains_nans_std(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.nan*np.ones((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], CorrStats())
        dv_av, std = monitor.average_components([dv0, dv1], True)
        np.testing.assert_array_equal(std, 0)
        self.assertTrue(np.all(dv0.sim_mat == dv_av.sim_mat))

    def test_result_std(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.random.random((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], CorrStats())
        dv_av, std = monitor.average_components([dv0, dv1], True)
        np.testing.assert_allclose(
            np.mean([dv0.sim_mat, dv1.sim_mat], axis=0), dv_av.sim_mat)
        np.testing.assert_allclose(
            np.std([dv0.sim_mat, dv1.sim_mat], axis=0), std)


if __name__ == "__main__":
    unittest.main()
