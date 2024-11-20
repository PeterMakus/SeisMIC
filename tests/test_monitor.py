'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 6th July 2021 09:18:14 am
Last Modified: Wednesday, 19th June 2024 10:39:35 am
'''

import os
import unittest
from unittest import mock
from unittest.mock import patch
import warnings
from copy import deepcopy

import numpy as np
from obspy import UTCDateTime

from seismic.monitor import monitor
from seismic.monitor.dv import DV
from seismic.correlate.stats import CorrStats
from seismic.db.corr_hdf5 import h5_FMTSTR


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
        self.p = os.path.curdir
        return super().setUp()

    @patch('seismic.monitor.monitor.glob')
    def test_no_match_str(self, glob_mock):
        glob_mock.return_value = [
            h5_FMTSTR.format(
                dir=self.p, network='b-b', station='a-a', location='-',
                channel='HHZ-HHZ'),
            h5_FMTSTR.format(
                dir=self.p, network='a-a', station='b-b', location='-',
                channel='HHZ-HHZ')]
        net = {
            'network': 'AA',
            'station': 'BB',
            'component': '*'
        }
        for ii in monitor.corr_find_filter(self.p, net):
            self.assertFalse(len(ii))

    @patch('seismic.monitor.monitor.glob')
    def test_no_match_list0(self, glob_mock):
        glob_mock.return_value = [
            h5_FMTSTR.format(
                dir=self.p, network='b-b', station='a-a', location='-',
                channel='HHZ-HHZ'),
            h5_FMTSTR.format(
                dir=self.p, network='a-a', station='b-b', location='-',
                channel='HHZ-HHZ')]
        net = {
            'network': 'AA',
            'station': ['AA', 'BB'],
            'component': 'Z'
        }
        for ii in monitor.corr_find_filter('.', net):
            self.assertFalse(len(ii))

    @patch('seismic.monitor.monitor.glob')
    def test_no_match_list1(self, glob_mock):
        glob_mock.return_value = [
            h5_FMTSTR.format(
                dir=self.p, network='b-b', station='a-a', location='-',
                channel='HHZ-HHZ'),
            h5_FMTSTR.format(
                dir=self.p, network='a-a', station='b-b', location='-',
                channel='HHZ-HHZ')]
        net = {
            'network': ['AA', 'BB'],
            'station': ['AA', 'BB'],
            'component': 'E'
        }
        for ii in monitor.corr_find_filter('.', net):
            self.assertFalse(len(ii))

    @patch('seismic.monitor.monitor.glob')
    def test_no_match_list_comp(self, glob_mock):
        glob_mock.return_value = [
            h5_FMTSTR.format(
                dir=self.p, network='b-b', station='a-a', location='-',
                channel='HHZ-HHZ'),
            h5_FMTSTR.format(
                dir=self.p, network='a-a', station='b-b', location='-',
                channel='HHZ-HHZ')]
        net = {
            'network': ['a', 'b'],
            'station': ['b', 'a'],
            'component': 'E'
        }
        for ii in monitor.corr_find_filter('.', net):
            self.assertFalse(len(ii))

    @patch('seismic.monitor.monitor.glob')
    def test_match_wildcard0(
            self, glob_mock: mock.MagicMock):
        glob_mock.return_value = [
            h5_FMTSTR.format(
                dir=self.p, network='b-b', station='a-a', location='-',
                channel='HHZ-HHZ'),
            h5_FMTSTR.format(
                dir=self.p, network='a-a', station='b-b', location='-',
                channel='HHZ-HHZ')]
        net = {
            'network': '*',
            'station': '*',
            'component': '*'
        }
        n, s, i = monitor.corr_find_filter(self.p, net)
        glob_mock.assert_called_once_with(
            h5_FMTSTR.format(
                dir=self.p, network='*', station='*', location='*',
                channel='*')
        )

        self.assertListEqual(['b-b', 'a-a'], n)
        self.assertListEqual(['a-a', 'b-b'], s)
        self.assertListEqual(
            [
                h5_FMTSTR.format(
                    dir=self.p, network='b-b', station='a-a', location='-',
                    channel='HHZ-HHZ'),
                h5_FMTSTR.format(
                    dir=self.p, network='a-a', station='b-b', location='-',
                    channel='HHZ-HHZ')], i)

    @patch('seismic.monitor.monitor.glob')
    def test_match_wildcard1(self, glob_mock):
        glob_mock.return_value = [
            h5_FMTSTR.format(
                dir=self.p, network='b-b', station='a-a', location='0-0',
                channel='HHZ-HHZ'),
            h5_FMTSTR.format(
                dir=self.p, network='a-a', station='b-b', location='0-0',
                channel='HHZ-HHZ')]
        net = {
            'network': 'a',
            'station': '*',
            'component': 'Z'
        }
        n, s, i = monitor.corr_find_filter(self.p, net)
        self.assertListEqual(['a-a'], n)
        self.assertListEqual(['b-b'], s)
        self.assertListEqual(
            [
                h5_FMTSTR.format(
                    dir=self.p, network='a-a', station='b-b', location='0-0',
                    channel='HHZ-HHZ')], i)

    @patch('seismic.monitor.monitor.glob')
    def test_match_list0(self, glob_mock):
        glob_mock.return_value = [
            h5_FMTSTR.format(
                dir=self.p, network='b-b', station='a-a', location='-',
                channel='HHZ-HHZ'),
            h5_FMTSTR.format(
                dir=self.p, network='a-a', station='b-b', location='0-0',
                channel='HHZ-HHZ')]
        net = {
            'network': ['a', 'b'],
            'station': ['a', 'b'],
            'component': '*'
        }
        n, s, i = monitor.corr_find_filter(os.path.curdir, net)

        self.assertListEqual(['b-b', 'a-a'], n)
        self.assertListEqual(['a-a', 'b-b'], s)
        self.assertListEqual(
            [
                h5_FMTSTR.format(
                    dir=self.p, network='b-b', station='a-a', location='-',
                    channel='HHZ-HHZ'),
                h5_FMTSTR.format(
                    dir=self.p, network='a-a', station='b-b', location='0-0',
                    channel='HHZ-HHZ')], i)

    @patch('seismic.monitor.monitor.glob')
    def test_match_list1(self, glob_mock):
        glob_mock.return_value = [
            h5_FMTSTR.format(
                dir=self.p, network='b-b', station='a-a', location='0-0',
                channel='HHE-HHZ'),
            h5_FMTSTR.format(
                dir=self.p, network='a-a', station='b-b', location='0-0',
                channel='HHN-HHZ')]
        net = {
            'network': 'a',
            'station': ['a', 'b'],
            'component': '*'
        }
        n, s, i = monitor.corr_find_filter(os.path.curdir, net)

        self.assertListEqual(['a-a'], n)
        self.assertListEqual(['b-b'], s)
        self.assertListEqual(
            [
                h5_FMTSTR.format(
                    dir=self.p, network='a-a', station='b-b', location='0-0',
                    channel='HHN-HHZ')], i)


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
        av_comp_mock.return_value = dv2
        av_dv = monitor.average_dvs_by_coords(
            [dv, dv2], lat, lon, return_std=True)
        av_comp_mock.assert_called_once_with([dv2], True)
        s = av_dv.stats
        np.testing.assert_array_equal([s.network, s.station], 'geoav')
        np.testing.assert_array_equal([s.stel, s.evel], 2*[(-1e6, 1e6)])
        np.testing.assert_array_equal([s.stlo, s.evlo], 2*[lon])
        np.testing.assert_array_equal([s.stla, s.evla], 2*[lat])

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
        av_dv = monitor.average_dvs_by_coords(
            [dv, dv2], lat, lon, el=(-100, 100), return_std=False)
        av_comp_mock.assert_called_once_with([dv2], False)
        s = av_dv.stats
        np.testing.assert_array_equal([s.network, s.station], 'geoav')
        np.testing.assert_array_equal([s.stel, s.evel], 2*[(-100, 100)])
        np.testing.assert_array_equal([s.stlo, s.evlo], 2*[lon])
        np.testing.assert_array_equal([s.stla, s.evla], 2*[lat])

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
            av_dv = monitor.average_dvs_by_coords(
                [dv, dv2], lat, lon, el=(-100, 100), return_std=False)
            self.assertEqual(len(w), 1)
        av_comp_mock.assert_called_once_with([dv2], False)
        s = av_dv.stats
        np.testing.assert_array_equal([s.network, s.station], 'geoav')
        np.testing.assert_array_equal([s.stel, s.evel], 2*[(-100, 100)])
        np.testing.assert_array_equal([s.stlo, s.evlo], 2*[lon])
        np.testing.assert_array_equal([s.stla, s.evla], 2*[lat])
        self.assertIsNone(av_dv.stretches)
        self.assertIsNone(av_dv.corrs)


class TestCorrectDVShift(unittest.TestCase):
    def setUp(self):
        self.dv = DV(
            np.arange(5)/5, np.linspace(0, 4, 5), 'bla',
            np.random.random((5, 5)),
            np.arange(5), 'dd', CorrStats())
        self.dv.value = self.dv.second_axis[
            np.nanargmax(np.nan_to_num(self.dv.sim_mat), axis=1)]
        self.dv.stats.corr_start = [UTCDateTime(ii) for ii in range(5)]

    def test_no_shift(self):
        dv = deepcopy(self.dv)
        monitor.correct_dv_shift(dv, self.dv)
        np.testing.assert_array_equal(self.dv.sim_mat, dv.sim_mat)
        np.testing.assert_array_equal(self.dv.value, dv.value)

    def test_no_shift_median_beyond(self):
        dv = deepcopy(self.dv)
        monitor.correct_dv_shift(dv, self.dv, method='median', n_overlap=5)
        np.testing.assert_array_equal(self.dv.sim_mat, dv.sim_mat)
        np.testing.assert_array_equal(self.dv.value, dv.value)

    def test_shift_by_1(self):
        dv = deepcopy(self.dv)
        # add dc offest
        dv.value += 1
        monitor.correct_dv_shift(dv, self.dv)
        np.testing.assert_array_equal(
            np.roll(self.dv.sim_mat, (-1, 0)), dv.sim_mat)


class TestCorrectTimeShiftSeveral(unittest.TestCase):
    def setUp(self):
        self.dv = DV(
            np.zeros(5), np.zeros(5), 'bla', np.zeros((5, 5)), np.zeros(5),
            'dd', CorrStats())

    @mock.patch('seismic.monitor.monitor.correct_dv_shift')
    def test_one_start(self, cts_mock: mock.MagicMock):
        dv = deepcopy(self.dv)
        dv.stats.corr_start = [UTCDateTime(ii) for ii in range(5)]
        monitor.correct_time_shift_several([dv], 'mean', 0)
        cts_mock.assert_not_called()

    @mock.patch('seismic.monitor.monitor.correct_dv_shift')
    def test_one_start2(self, cts_mock: mock.MagicMock):
        dv = deepcopy(self.dv)
        dv.stats.corr_start = [UTCDateTime(ii) for ii in range(5)]
        dv2 = dv
        monitor.correct_time_shift_several([dv, dv2], 'mean', 0)
        cts_mock.assert_not_called()

    @mock.patch('seismic.monitor.monitor.correct_dv_shift')
    def test_sort_and_shift(self, cts_mock: mock.MagicMock):
        dv0 = deepcopy(self.dv)
        dv1 = deepcopy(self.dv)
        dv2 = deepcopy(self.dv)
        for k, dv in enumerate([dv0, dv1, dv2]):
            dv.stats.corr_start = [UTCDateTime(ii+k) for ii in range(5)]
        monitor.correct_time_shift_several([dv1, dv0, dv2], 'mean', 0)
        calls = [
            mock.call(dv1, dv0, method='mean', n_overlap=0),
            mock.call(dv2, dv1, method='mean', n_overlap=0)]
        cts_mock.assert_has_calls(calls)

    @mock.patch('seismic.monitor.monitor.correct_dv_shift')
    def test_except(self, cts_mock: mock.MagicMock):
        dv0 = deepcopy(self.dv)
        dv1 = deepcopy(self.dv)
        for k, dv in enumerate([dv0, dv1]):
            dv.stats.corr_start = [UTCDateTime(ii+k) for ii in range(5)]
        cts_mock.side_effect = [ValueError]
        with warnings.catch_warnings(record=True) as w:
            monitor.correct_time_shift_several([dv1, dv0], 'mean', 0)
        self.assertEqual(len(w), 1)
        cts_mock.assert_called_once_with(dv1, dv0, method='mean', n_overlap=0)


class TestAverageComponents(unittest.TestCase):
    def test_differing_shape(self):
        sim0 = np.zeros((5, 5))
        sim1 = np.zeros((6, 6))
        corr0 = np.zeros((5))
        corr1 = np.zeros((6))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr1, corr1, ['stretch'], sim1, corr1, ['bla'], CorrStats())
        with self.assertWarns(UserWarning):
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
        with self.assertWarns(UserWarning):
            monitor.average_components([dv0, dv1])

    @mock.patch('seismic.monitor.monitor.correct_time_shift_several')
    def test_contains_nans(self, ctss_mock: mock.MagicMock):
        sim0 = np.random.random((5, 5))
        sim1 = np.nan*np.ones((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], CorrStats())
        dv_av = monitor.average_components(
            [dv0, dv1], correct_shift=True, correct_shift_method='bla',
            correct_shift_overlap=5)
        ctss_mock.assert_called_once_with(
            [dv0, dv1], method='bla', n_overlap=5)
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

    def test_header(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.random.random((5, 5))
        corr0 = np.zeros((5))
        stats0 = CorrStats()
        stats0['network'] = stats0['station'] = stats0['channel'] = 'A'
        stats1 = CorrStats()
        stats1['network'] = stats1['station'] = stats1['channel'] = 'B'
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], stats0)
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], stats1)
        dv_av = monitor.average_components([dv0, dv1])
        self.assertEqual(dv_av.stats.station, 'av')
        self.assertEqual(dv_av.stats.network, 'av')
        self.assertEqual(dv_av.stats.channel, 'av')

    def test_header2(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.random.random((5, 5))
        corr0 = np.zeros((5))
        stats0 = CorrStats()
        stats0['network'] = stats0['station'] = stats0['channel'] = 'A'
        stats1 = CorrStats()
        stats1['network'] = stats1['station'] = stats1['channel'] = 'A'
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], stats0)
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], stats1)
        dv_av = monitor.average_components([dv0, dv1])
        self.assertEqual(dv_av.stats.station, 'A')
        self.assertEqual(dv_av.stats.network, 'A')
        self.assertEqual(dv_av.stats.channel, 'A')


class TestAverageComponentsMemSave(unittest.TestCase):
    def test_differing_shape(self):
        sim0 = np.zeros((5, 5))
        sim1 = np.zeros((6, 6))
        corr0 = np.zeros((5))
        corr1 = np.zeros((6))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr1, corr1, ['stretch'], sim1, corr1, ['bla'], CorrStats())
        with self.assertWarns(UserWarning):
            monitor.average_components_mem_save([dv0, dv1])

    def test_differing_methods(self):
        sim0 = np.zeros((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['blub'], CorrStats())
        with self.assertRaises(TypeError):
            monitor.average_components_mem_save([dv0, dv1])

    def test_differing_2ndax(self):
        sim0 = np.zeros((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(
            corr0, corr0, ['stretch'], sim0, corr0+1, ['bla'], CorrStats())
        with self.assertWarns(UserWarning):
            monitor.average_components_mem_save([dv0, dv1])

    def test_contains_nans(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.nan*np.ones((5, 5))
        corr0 = np.zeros((5))
        corr1 = np.zeros((5)) + np.nan
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr1, corr1, ['stretch'], sim1, corr0, ['bla'], CorrStats())
        dv_av = monitor.average_components_mem_save([dv0, dv1])
        np.testing.assert_array_equal(dv0.sim_mat, dv_av.sim_mat)

    def test_result(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.random.random((5, 5))
        corr0 = np.zeros((5))
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], CorrStats())
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], CorrStats())
        dv_av = monitor.average_components_mem_save([dv0, dv1])
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
            dv_av = monitor.average_components_mem_save([dv0, dv1])
            self.assertEqual(len(w), 1)
        np.testing.assert_array_equal(dv1.sim_mat, dv_av.sim_mat)

    def test_header(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.random.random((5, 5))
        corr0 = np.zeros((5))
        stats0 = CorrStats()
        stats0['network'] = stats0['station'] = stats0['channel'] = 'A'
        stats1 = CorrStats()
        stats1['network'] = stats1['station'] = stats1['channel'] = 'B'
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], stats0)
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], stats1)
        dv_av = monitor.average_components_mem_save([dv0, dv1])
        self.assertEqual(dv_av.stats.station, 'av')
        self.assertEqual(dv_av.stats.network, 'av')
        self.assertEqual(dv_av.stats.channel, 'av')

    def test_header2(self):
        sim0 = np.random.random((5, 5))
        sim1 = np.random.random((5, 5))
        corr0 = np.zeros((5))
        stats0 = CorrStats()
        stats0['network'] = stats0['station'] = stats0['channel'] = 'A'
        stats1 = CorrStats()
        stats1['network'] = stats1['station'] = stats1['channel'] = 'A'
        dv0 = DV(corr0, corr0, ['stretch'], sim0, corr0, ['bla'], stats0)
        dv1 = DV(corr0, corr0, ['stretch'], sim1, corr0, ['bla'], stats1)
        dv_av = monitor.average_components_mem_save([dv0, dv1])
        self.assertEqual(dv_av.stats.station, 'A')
        self.assertEqual(dv_av.stats.network, 'A')
        self.assertEqual(dv_av.stats.channel, 'A')


if __name__ == "__main__":
    unittest.main()
