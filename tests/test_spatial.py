'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 16th January 2023 11:07:27 am
Last Modified: Tuesday, 5th March 2024 02:55:54 pm
'''

import unittest
from unittest import mock
import warnings
import os
from copy import deepcopy

from obspy import UTCDateTime
import numpy as np

from seismic.monitor import spatial as spt
from seismic.monitor.dv import DV
from seismic.correlate.stats import CorrStats


class TestProbability(unittest.TestCase):
    def test_no_chance(self):
        # distance larger than t*v
        dist = np.random.rand() + 3
        t = np.random.rand() + 5
        c = dist/t - np.random.rand()
        # mf path should not matter
        self.assertEqual(spt.probability(dist, t, c, 10, 1e-6), 0)

    def test_vel0(self):
        with self.assertRaises(ZeroDivisionError):
            spt.probability(1, 1, 0, 10, 1e-9), 1

    def test_ballistic(self):
        # distance=t*v
        dist = 3
        t = 5
        c = dist/t
        # mf path should not matter
        self.assertGreater(spt.probability(dist, t, c, 10, 1e-6), 0)

    def test_ballistic_array(self):
        # distance=t*v
        dist = np.ones(40)*2
        t = 5
        c = 2/t
        # mf path should not matter
        np.testing.assert_array_less(
            0, spt.probability(dist, t, c, 10, 1e-6))

    def test_ballistic_array2(self):
        dist = np.ones(40)*2
        t = np.arange(6)
        c = 2/5
        # mf path should not matter
        prob = spt.probability(dist, t, c, 10, 1e-6)
        np.testing.assert_array_less(0, prob[-1, :, :])
        # before the ballistic arrival, the probability should be 0
        np.testing.assert_array_almost_equal(prob[0:-1], 0)

    def test_prob_rand(self):
        # in any other case this should be 0 < probability < 1
        dist = 3
        t = 5 + np.random.rand()
        c = 2/t + .01
        # mf path should not matter
        self.assertGreater(spt.probability(dist, t, c, 10, c*t/2), 0)
        self.assertLess(spt.probability(dist, t, c, 10, c*t/2), 1)

    def test_dist_neg(self):
        dist = -3
        t = 5
        c = dist/t
        with self.assertRaises(ValueError):
            spt.probability(dist, t, c, 10, c*t/2)

    def test_t_neg(self):
        dist = 3
        t = -1
        c = 1
        with self.assertRaises(ValueError):
            spt.probability(dist, t, c, 10, 1e-3)


class TestComputeGridDist(unittest.TestCase):
    def test_different_size(self):
        X = np.zeros(5)
        Y = np.arange(10)
        d = spt.compute_grid_dist(X, Y, 0, 0)
        np.testing.assert_equal(np.reshape(np.tile(Y, 5), (5, 10)).T, d)

    def test_same_size(self):
        X = np.random.rand((5))*-5
        Y = np.ones(5)
        d = spt.compute_grid_dist(X, Y, 0, 1)
        np.testing.assert_almost_equal(
            abs(np.reshape(np.tile(X, 5), (5, 5))), d)


class TestSensitivityKernel(unittest.TestCase):
    @mock.patch('seismic.monitor.spatial.compute_grid_dist')
    @mock.patch('seismic.monitor.spatial.probability')
    def test_result(self, pbb_mock, cgd_mock):
        dt = 1
        s1 = s2 = np.array([5, 5])
        x = y = np.arange(10)
        t = 20
        vel = mf_path = 1
        cgd_mock.return_value = np.zeros((10, 10))
        pbb_mock.side_effect = [1] + [np.ones((10, 10))]*50
        K = spt.sensitivity_kernel(s1, s2, x, y, t, dt, vel, mf_path)
        cgd_calls = [
            mock.call(x, y, s1[0], s1[1]),
            mock.call(x, y, s2[0], s2[1])
        ]
        cgd_mock.assert_has_calls(cgd_calls)
        # To be continued here
        self.assertEqual(pbb_mock.call_count, 43)

        calls = [mock.call(0, 20, 1, 1, atol=dt*vel/2)]
        for tt in np.arange(0, t + dt, dt):
            calls.extend([
                mock.call(mock.ANY, tt, vel, mf_path, atol=dt*vel/2),
                mock.call(mock.ANY, t-tt, vel, mf_path, atol=dt*vel/2)])
        pbb_mock.assert_has_calls(calls)

        np.testing.assert_almost_equal(K, np.ones((10, 10))*t)

    @mock.patch('seismic.monitor.spatial.compute_grid_dist')
    @mock.patch('seismic.monitor.spatial.probability')
    def test_dstat_too_large(self, pbb_mock, cgd_mock):
        dt = 1
        s1 = np.array([5, 5])
        s2 = np.array([5, 25])
        x = y = np.arange(10)
        t = 10
        vel = mf_path = 1
        cgd_mock.return_value = np.zeros((10, 10))
        pbb_mock.side_effect = [0] + [np.ones((10, 10))]*50
        K = spt.sensitivity_kernel(s1, s2, x, y, t, dt, vel, mf_path)
        cgd_calls = [
            mock.call(x, y, s1[0], s1[1])]
        cgd_mock.assert_has_calls(cgd_calls)
        # To be continued here
        self.assertEqual(pbb_mock.call_count, 1)
        pbb_mock.assert_called_once_with(20.0, 10, 1, 1, atol=dt*vel/2)

        np.testing.assert_array_equal(K, 0)

    def test_x_not_monotoneous(self):
        dt = .1
        s1 = s2 = np.array([5, 5])
        x = np.random.rand(10)
        y = np.arange(10)
        t = 20
        vel = mf_path = 1
        with self.assertRaises(ValueError):
            spt.sensitivity_kernel(s1, s2, x, y, t, dt, vel, mf_path)

    def test_y_not_monotoneous(self):
        dt = .1
        s1 = s2 = np.array([5, 5])
        y = np.random.rand(10)
        x = np.arange(10)
        t = 20
        vel = mf_path = 1
        with self.assertRaises(ValueError):
            spt.sensitivity_kernel(s1, s2, x, y, t, dt, vel, mf_path)

    def test_dt_neg(self):
        dt = -.1
        s1 = s2 = np.array([5, 5])
        x = y = np.arange(10)
        t = 20
        vel = mf_path = 1
        with self.assertRaises(ValueError):
            spt.sensitivity_kernel(s1, s2, x, y, t, dt, vel, mf_path)

    @mock.patch('seismic.monitor.spatial.compute_grid_dist')
    @mock.patch('seismic.monitor.spatial.probability')
    def test_warn_on_large_dt(self, pbb_mock, cgd_mock):
        dt = 1
        s1 = s2 = np.array([5, 5])
        x = y = np.arange(10)
        t = 20
        vel = mf_path = 1
        cgd_mock.return_value = np.zeros((10, 10))
        pbb_mock.side_effect = [1] + [np.ones((10, 10))]*50
        with warnings.catch_warnings(record=True) as w:
            spt.sensitivity_kernel(s1, s2, x, y, t, dt, vel, mf_path)
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, UserWarning))


class TestDataVariance(unittest.TestCase):
    def test_result(self):
        corr = np.ones(10)
        bw = 2
        tw = (5, 10)
        freq_c = 3
        out = spt.data_variance(corr, bw, tw, freq_c)
        np.testing.assert_equal(np.zeros(10), out)

    def test_result2(self):
        # All allowed values so no warnings should occur
        corr = (np.arange(10) + 1)/10
        bw = 2
        tw = (5, 10)
        freq_c = 3
        with warnings.catch_warnings(record=True) as w:
            spt.data_variance(corr, bw, tw, freq_c)
        self.assertEqual(len(w), 0)

    def test_raises_error_for_neg_tw(self):
        corr = np.ones(10)
        bw = 2
        tw = (-5, 10)
        freq_c = 3
        with self.assertRaises(ValueError):
            spt.data_variance(corr, bw, tw, freq_c)

    def test_raises_error_for_neg_bw(self):
        corr = np.ones(10)
        bw = -2
        tw = (5, 10)
        freq_c = 3
        with self.assertRaises(ValueError):
            spt.data_variance(corr, bw, tw, freq_c)

    def test_raises_error_for_neg_freqc(self):
        corr = np.ones(10)
        bw = 2
        tw = (5, 10)
        freq_c = -3
        with self.assertRaises(ValueError):
            spt.data_variance(corr, bw, tw, freq_c)

    def test_raise_error_for_tw1_greater_than_tw0(self):
        corr = np.ones(10)
        bw = 2
        tw = (10, 5)
        freq_c = 3
        with self.assertRaises(ValueError):
            spt.data_variance(corr, bw, tw, freq_c)


class TestComputeCM(unittest.TestCase):
    def test_result(self):
        sf = cl = std = 1
        dist = np.zeros(5)
        np.testing.assert_allclose(
            spt.compute_cm(sf, cl, std, dist), np.ones(5))


class TestGeo2Cart(unittest.TestCase):
    def test_result_latlonpos(self):
        lat = np.arange(10)
        lon = np.arange(11)
        lat0 = 0
        lon0 = 0
        out = spt.geo2cart(lat, lon, lat0, lon0)
        self.assertEqual(out[0].shape, lon.shape)
        self.assertEqual(out[1].shape, lat.shape)
        self.assertTrue(np.all(out[0] >= 0))
        self.assertTrue(np.all(out[1] >= 0))

    def test_corner_inside(self):
        lat = np.arange(10)
        lon = np.arange(11)
        lat0 = 1
        lon0 = 0
        with self.assertRaises(ValueError):
            spt.geo2cart(lat, lon, lat0, lon0)

    def test_result_len1(self):
        # Just making sure that it still remains a np array
        # otherwise this would introduce bugs
        lat = np.arange(1)
        lon = np.arange(-10, -9, 1)
        lon0 = -11
        lat0 = -5
        out = spt.geo2cart(lat, lon, lat0, lon0)
        self.assertEqual(out[0].shape, lon.shape)
        self.assertEqual(out[1].shape, lat.shape)
        self.assertTrue(np.all(out[1] >= 0))


class TestDVGrid(unittest.TestCase):
    def setUp(self):
        dt = vel = mf_path = 1
        self.dvg = spt.DVGrid(0, 0, 1, 9, 9, dt, vel, mf_path)

    def test_forward_model_from_dv(self):
        mod = np.ones((10, 10))
        dvmock = mock.MagicMock()
        with (
            mock.patch.object(self.dvg, '_extract_info_dvs') as ex_mock,
            mock.patch.object(
                self.dvg, '_compute_sensitivity_kernels') as st_mock
        ):
            ex_mock.return_value = (
                None, None, 'slat0', 'slon0', 'slat1', 'slon1', (3, 7), None,
                None)
            st_mock.return_value, _ = np.meshgrid(
                np.arange(100), np.arange(100))
            out = self.dvg.forward_model(mod, dvmock, utc=0)
            ex_mock.assert_called_once_with(dvmock, 0, False)
            st_mock.assert_called_once_with(
                'slat0', 'slon0', 'slat1', 'slon1', 5)
        np.testing.assert_allclose(out, np.ones(100)*4950)

    def test_forward_model_from_kwargs(self):
        mod = np.ones((10, 10))
        with mock.patch.object(
                self.dvg, '_compute_sensitivity_kernels') as st_mock:
            st_mock.return_value, _ = np.meshgrid(
                np.arange(100), np.arange(100))
            out = self.dvg.forward_model(
                mod, tw=(3, 7), stat0=[('slat0', 'slon0')],
                stat1=[('slat1', 'slon1')])
            st_mock.assert_called_once_with(
                ('slat0',), ('slon0',), ('slat1',), ('slon1',), 5)
        np.testing.assert_allclose(out, np.ones(100)*4950)

    def test_wrong_grid(self):
        mod = np.ones((100, 100))
        dvmock = mock.MagicMock()
        with self.assertRaises(ValueError):
            self.dvg.forward_model(mod, dvmock, utc=0)

    def test_missing_args(self):
        mod = np.ones((10, 10))
        with self.assertRaises(TypeError):
            self.dvg.forward_model(mod)

    @mock.patch('seismic.monitor.spatial.compute_cm')
    def test_compute_dv_grid_from_dv(self, cm_mock: mock.MagicMock):
        cm_mock.return_value = np.zeros((100, 100))

        scaling_factor, corr_len, std_model = [4, 5, 1]
        dvmock = mock.MagicMock()
        with (
            mock.patch.object(self.dvg, '_extract_info_dvs') as ex_mock,
            mock.patch.object(
                self.dvg, '_compute_sensitivity_kernels') as st_mock,
            mock.patch.object(self.dvg, '_compute_dist_matrix') as dm_mock,
            mock.patch.object(self.dvg, '_compute_cd') as cd_mock
        ):
            ex_mock.return_value = (
                np.zeros(50), np.ones(50), 'slat0', 'slon0', 'slat1',
                'slon1', (3, 7), 3, 5)
            st_mock.return_value = np.ones((50, 100)) - .5
            dm_mock.return_value, _ = np.meshgrid(
                np.arange(100), np.arange(100))
            cd_mock.return_value = np.eye(50)

            out = self.dvg.compute_dv_grid(
                dvmock, 0, scaling_factor, corr_len,
                std_model, compute_resolution=False)
            ex_mock.assert_called_once_with(dvmock, 0, False)
            st_mock.assert_called_once_with(
                'slat0', 'slon0', 'slat1', 'slon1', 5)
            cd_mock.assert_called_once_with(
                mock.ANY, 3, 5, (3, 7), mock.ANY)
        cm_mock.assert_called_once_with(
            scaling_factor, corr_len, std_model, mock.ANY)
        np.testing.assert_allclose(out, np.zeros((10, 10)))

    @mock.patch('seismic.monitor.spatial.compute_cm')
    def test_compute_dv_grid_from_kwargs(self, cm_mock: mock.MagicMock):
        cm_mock.return_value = np.zeros((100, 100))

        scaling_factor, corr_len, std_model = [4, 5, 1]
        dvmock = mock.MagicMock()
        with (
            mock.patch.object(self.dvg, '_extract_info_dvs') as ex_mock,
            mock.patch.object(
                self.dvg, '_compute_sensitivity_kernels') as st_mock,
            mock.patch.object(self.dvg, '_compute_dist_matrix') as dm_mock,
            mock.patch.object(self.dvg, '_compute_cd') as cd_mock,
            mock.patch.object(self.dvg, '_compute_resolution') as res_mock
        ):
            ex_mock.return_value = (
                np.zeros(50), np.ones(50), 'slat0', 'slon0', 'slat1',
                'slon1', None, None, None)
            st_mock.return_value = np.ones((50, 100)) - .5
            dm_mock.return_value, _ = np.meshgrid(
                np.arange(100), np.arange(100))
            cd_mock.return_value = np.eye(50)

            out = self.dvg.compute_dv_grid(
                dvmock, 0, scaling_factor, corr_len,
                std_model, compute_resolution=True, tw=(3, 7), freq0=3,
                freq1=5)
            ex_mock.assert_called_once_with(dvmock, 0, False)
            st_mock.assert_called_once_with(
                'slat0', 'slon0', 'slat1', 'slon1', 5)
            cd_mock.assert_called_once_with(
                mock.ANY, 3, 5, (3, 7), mock.ANY)
            res_mock.assert_called_once()
        cm_mock.assert_called_once_with(
            scaling_factor, corr_len, std_model, mock.ANY)
        np.testing.assert_allclose(out, np.zeros((10, 10)))

    def test_compute_dv_grid_dvproc_missing(self):
        scaling_factor, corr_len, std_model = [4, 5, 1]
        dvmock = mock.MagicMock()
        with (
            mock.patch.object(self.dvg, '_extract_info_dvs') as ex_mock,
            self.assertRaises(AttributeError)
        ):
            ex_mock.return_value = (
                np.zeros(50), np.ones(50), 'slat0', 'slon0', 'slat1',
                'slon1', None, None, None)
            self.dvg.compute_dv_grid(
                dvmock, 0, scaling_factor, corr_len,
                std_model, compute_resolution=False)
            ex_mock.assert_called_once_with(dvmock, 0)

    @mock.patch('seismic.monitor.spatial.compute_cm')
    def test_compute_posterior_cov(self, cm_mock: mock.MagicMock):
        cm_mock.return_value = np.eye(100)

        scaling_factor, corr_len, std_model = [4, 5, 1]
        dvmock = mock.MagicMock()
        with (
            mock.patch.object(self.dvg, '_extract_info_dvs') as ex_mock,
            mock.patch.object(
                self.dvg, '_compute_sensitivity_kernels') as st_mock,
            mock.patch.object(self.dvg, '_compute_dist_matrix') as dm_mock,
            mock.patch.object(self.dvg, '_compute_cd') as cd_mock,
            mock.patch.object(self.dvg, '_compute_posterior_covariance') as
                pc_mock
        ):
            ex_mock.return_value = (
                np.zeros(50), np.ones(50), 'slat0', 'slon0', 'slat1',
                'slon1', (3, 7), 3, 5)
            st_mock.return_value = np.ones((50, 100)) - .5
            dm_mock.return_value, _ = np.meshgrid(
                np.arange(100), np.arange(100))
            cd_mock.return_value = np.eye(50)
            pc_mock.return_value = np.zeros((50, 50))

            out = self.dvg.compute_posterior_covariance(
                dvmock, 0, scaling_factor, corr_len, std_model)
            ex_mock.assert_called_once_with(dvmock, 0, False)
            st_mock.assert_called_once_with(
                'slat0', 'slon0', 'slat1', 'slon1', 5)
            cd_mock.assert_called_once_with(
                mock.ANY, 3, 5, (3, 7), mock.ANY)
        cm_mock.assert_called_once_with(
            scaling_factor, corr_len, std_model, mock.ANY)
        np.testing.assert_allclose(out, np.zeros((50, 50)))

    @mock.patch('seismic.monitor.spatial.compute_cm')
    def test_compute_posterior_cov_no_info(
            self, cm_mock: mock.MagicMock):
        cm_mock.return_value = np.eye(100)

        scaling_factor, corr_len, std_model = [4, 5, 1]
        dvmock = mock.MagicMock()
        with (
            mock.patch.object(self.dvg, '_extract_info_dvs') as ex_mock,
            self.assertRaises(AttributeError)
        ):
            ex_mock.return_value = (
                np.zeros(50), np.ones(50), 'slat0', 'slon0', 'slat1',
                'slon1', None, None, None)
            self.dvg.compute_posterior_covariance(
                dvmock, 0, scaling_factor, corr_len, std_model)
            ex_mock.assert_called_once_with(dvmock, 0, False)
        cm_mock.assert_called_once_with(
            scaling_factor, corr_len, std_model, mock.ANY)

    @mock.patch('seismic.monitor.spatial.compute_cm')
    def test_compute_res_from_dv(self, cm_mock: mock.MagicMock):
        cm_mock.return_value = np.zeros((100, 100))

        scaling_factor, corr_len, std_model = [4, 5, 1]
        dvmock = mock.MagicMock()
        with (
            mock.patch.object(self.dvg, '_extract_info_dvs') as ex_mock,
            mock.patch.object(
                self.dvg, '_compute_sensitivity_kernels') as st_mock,
            mock.patch.object(self.dvg, '_compute_dist_matrix') as dm_mock,
            mock.patch.object(self.dvg, '_compute_cd') as cd_mock,
            mock.patch.object(self.dvg, '_compute_resolution')
        ):
            ex_mock.return_value = (
                np.zeros(50), np.ones(50), 'slat0', 'slon0', 'slat1',
                'slon1', (3, 7), 3, 5)
            st_mock.return_value = np.ones((50, 100)) - .5
            dm_mock.return_value, _ = np.meshgrid(
                np.arange(100), np.arange(100))
            cd_mock.return_value = np.eye(50)

            self.dvg.compute_resolution(
                dvmock, 0, scaling_factor, corr_len, std_model)
            ex_mock.assert_called_once_with(dvmock, 0, False)
            st_mock.assert_called_once_with(
                'slat0', 'slon0', 'slat1', 'slon1', 5)
            cd_mock.assert_called_once_with(
                mock.ANY, 3, 5, (3, 7), mock.ANY)
        cm_mock.assert_called_once_with(
            scaling_factor, corr_len, std_model, mock.ANY)

    @mock.patch('seismic.monitor.spatial.compute_cm')
    def test_compute_res_from_kwargs(self, cm_mock: mock.MagicMock):
        cm_mock.return_value = np.zeros((100, 100))

        scaling_factor, corr_len, std_model = [4, 5, 1]
        dvmock = mock.MagicMock()
        with (
            mock.patch.object(self.dvg, '_extract_info_dvs') as ex_mock,
            mock.patch.object(
                self.dvg, '_compute_sensitivity_kernels') as st_mock,
            mock.patch.object(self.dvg, '_compute_dist_matrix') as dm_mock,
            mock.patch.object(self.dvg, '_compute_cd') as cd_mock,
            mock.patch.object(self.dvg, '_compute_resolution') as res_mock
        ):
            ex_mock.return_value = (
                np.zeros(50), np.ones(50), 'slat0', 'slon0', 'slat1',
                'slon1', None, None, None)
            st_mock.return_value = np.ones((50, 100)) - .5
            dm_mock.return_value, _ = np.meshgrid(
                np.arange(100), np.arange(100))
            cd_mock.return_value = np.eye(50)
            self.dvg.compute_resolution(
                dvmock, 0, scaling_factor, corr_len, std_model, tw=(3, 7),
                freq0=3, freq1=5)
            ex_mock.assert_called_once_with(dvmock, 0, False)
            st_mock.assert_called_once_with(
                'slat0', 'slon0', 'slat1', 'slon1', 5)
            cd_mock.assert_called_once_with(
                mock.ANY, 3, 5, (3, 7), mock.ANY)
            res_mock.assert_called_once()
        cm_mock.assert_called_once_with(
            scaling_factor, corr_len, std_model, mock.ANY)

    def test_compute_res_dvproc_missing(self):
        scaling_factor, corr_len, std_model = [4, 5, 1]
        dvmock = mock.MagicMock()
        with (
            mock.patch.object(self.dvg, '_extract_info_dvs') as ex_mock,
            self.assertRaises(AttributeError)
        ):
            ex_mock.return_value = (
                np.zeros(50), np.ones(50), 'slat0', 'slon0', 'slat1',
                'slon1', None, None, None)
            self.dvg.compute_resolution(
                dvmock, 0, scaling_factor, corr_len, std_model)
            ex_mock.assert_called_once_with(dvmock, 0)

    def test_compute_res_int(self):
        skernels = np.ones((100, 100))
        b = np.eye(100)
        a = np.ones((100, 100))
        out = self.dvg._compute_resolution(skernels, a, b)
        np.testing.assert_almost_equal(out, 1e4*np.ones((10, 10)))

    def test_extract_dv(self):
        dv = mock.MagicMock()
        dv.value = np.arange(100)
        dv.corr = np.arange(100)/100
        dv.stats.corr_start = np.arange(100) + 100
        dv.stats.corr_end = dv.stats.corr_start + 1
        dv.stats.stla = 35.2
        dv.stats.stlo = -5
        dv.stats.evla = 34.2
        dv.stats.evlo = -6
        dv.dv_processing = {
            'tw_start': 3, 'tw_len': 4, 'freq_min': 1, 'freq_max': 3}
        with mock.patch.object(self.dvg, '_add_stations') as stat_mock:
            val, corr, slat0, slon0, slat1, slon1, tw, freq0, freq1\
                = self.dvg._extract_info_dvs([dv], 155, verbose=False)
            stat_mock.assert_has_calls([
                mock.call(np.array(35.2), np.array(-5)),
                mock.call(np.array(34.2), np.array(-6))]
            )
        self.assertEqual(val, 55)
        self.assertEqual(corr, .55)
        self.assertEqual(slat0, 35.2)
        self.assertEqual(slon0, -5)
        self.assertEqual(slat1, 34.2)
        self.assertEqual(slon1, -6)
        self.assertTupleEqual(tw, (3, 7))
        self.assertEqual(freq0, 1)
        self.assertEqual(freq1, 3)

    def test_extract_dv_outside_time(self):
        dv = mock.MagicMock()
        dv.value = np.arange(100)
        dv.corr = np.arange(100)/100
        dv.stats.corr_start = np.arange(100) + 100
        dv.stats.corr_end = dv.stats.corr_start + 1
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaises(IndexError):
                self.dvg._extract_info_dvs([dv], 1, verbose=True)
            self.assertEqual(len(w), 1)

    def test_extract_dv_no_proc(self):
        dv = mock.MagicMock()
        dv.value = np.arange(100)
        dv.corr = np.arange(100)/100
        dv.stats.corr_start = np.arange(100) + 100
        dv.stats.corr_end = dv.stats.corr_start + 1
        dv.stats.stla = 35.2
        dv.stats.stlo = -5
        dv.stats.evla = 34.2
        dv.stats.evlo = -6
        dv.dv_processing = None
        with mock.patch.object(self.dvg, '_add_stations') as stat_mock:
            val, corr, slat0, slon0, slat1, slon1, tw, freq0, freq1\
                = self.dvg._extract_info_dvs([dv], 155, verbose=True)
            stat_mock.assert_has_calls([
                mock.call(np.array(35.2), np.array(-5)),
                mock.call(np.array(34.2), np.array(-6))]
            )
        self.assertEqual(val, 55)
        self.assertEqual(corr, .55)
        self.assertEqual(slat0, 35.2)
        self.assertEqual(slon0, -5)
        self.assertEqual(slat1, 34.2)
        self.assertEqual(slon1, -6)
        self.assertIsNone(tw)
        self.assertIsNone(freq0)
        self.assertIsNone(freq1)

    def test_extract_dv_nan(self):
        dv = mock.MagicMock()
        dv.value = np.arange(100)
        dv.corr = np.arange(100)/100
        dv.stats.corr_start = np.arange(100) + 100
        dv.stats.corr_end = dv.stats.corr_start + 1
        dv.stats.stla = 35.2
        dv.stats.stlo = -5
        dv.stats.evla = 34.2
        dv.stats.evlo = -6
        dv.dv_processing = None
        dv2 = deepcopy(dv)
        dv2.value = dv2.value + np.nan
        dvs = [dv, dv2]
        with mock.patch.object(self.dvg, '_add_stations') as stat_mock:
            val, corr, slat0, slon0, slat1, slon1, tw, freq0, freq1\
                = self.dvg._extract_info_dvs(dvs, 155, verbose=False)
            stat_mock.assert_has_calls([
                mock.call(np.array(35.2), np.array(-5)),
                mock.call(np.array(34.2), np.array(-6))]
            )
        self.assertEqual(val, 55)
        self.assertEqual(corr, .55)
        self.assertEqual(slat0, 35.2)
        self.assertEqual(slon0, -5)
        self.assertEqual(slat1, 34.2)
        self.assertEqual(slon1, -6)
        self.assertIsNone(tw)
        self.assertIsNone(freq0)
        self.assertIsNone(freq1)

    def test_extract_dv_only_nan(self):
        dv = mock.MagicMock()
        dv.value = np.arange(100) + np.nan
        dv.corr = np.arange(100)/100
        dv.stats.corr_start = np.arange(100) + 100
        dv.stats.corr_end = dv.stats.corr_start + 1
        dv.stats.stla = 35.2
        dv.stats.stlo = -5
        dv.stats.evla = 34.2
        dv.stats.evlo = -6
        dv.dv_processing = None
        with self.assertRaises(IndexError):
            self.dvg._extract_info_dvs([dv], 155, verbose=False)

    def test_extract_dv_utc_outside(self):
        dv = mock.MagicMock()
        dv.value = np.arange(100) + np.nan
        dv.corr = np.arange(100)/100
        dv.stats.corr_start = np.arange(100) + 100
        dv.stats.corr_end = dv.stats.corr_start + 1
        dv.stats.stla = 35.2
        dv.stats.stlo = -5
        dv.stats.evla = 34.2
        dv.stats.evlo = -6
        dv.dv_processing = None
        with self.assertRaises(IndexError):
            self.dvg._extract_info_dvs([dv], 500, verbose=False)

    @mock.patch('seismic.monitor.spatial.geo2cart')
    def test_add_stations_no_given(self, g2c_mock):
        dvg = deepcopy(self.dvg)
        lats = np.arange(10)
        lons = np.arange(10) + 10
        g2c_mock.return_value = (lats, lons)
        dvg._add_stations(lats, lons)
        # Make sure that order is not messed up
        for xs, ys in zip(dvg.statx, dvg.staty):
            self.assertIn(xs, lats)
            self.assertIn(ys, lons)
            np.testing.assert_array_almost_equal(
                np.where(lats == xs), np.where(lons == ys)
            )

    @mock.patch('seismic.monitor.spatial.geo2cart')
    def test_add_stations_no_given_wdupl(self, g2c_mock):
        dvg = deepcopy(self.dvg)
        lats = np.arange(10)
        lons = np.arange(10) + 10
        g2c_mock.return_value = (np.ones(5), np.zeros(5))
        dvg._add_stations(lats, lons)
        # Make sure that order is not messed up
        self.assertEqual(dvg.statx, 1)
        self.assertEqual(dvg.staty, 0)

    @mock.patch('seismic.monitor.spatial.geo2cart')
    def test_add_stations_given(self, g2c_mock):
        dvg = deepcopy(self.dvg)
        lats = np.arange(10)
        lons = np.arange(10) + 10
        dvg.statx = lats + 5
        dvg.statlon = lons
        dvg.statlat = lats
        dvg.staty = lons + 5
        g2c_mock.return_value = (lats, lons)
        dvg._add_stations(lats, lons)
        # Make sure that order is not messed up
        latxp = np.arange(15)
        lonxp = np.arange(15) + 10
        for xs, ys in zip(dvg.statx, dvg.staty):
            self.assertIn(xs, latxp)
            self.assertIn(ys, lonxp)
            np.testing.assert_array_almost_equal(
                np.where(latxp == xs), np.where(lonxp == ys))

    @mock.patch('seismic.monitor.spatial.data_variance')
    def test_compute_cd(self, dv_mock: mock.MagicMock):
        dv_mock.return_value = np.arange(100)/100
        skernels = np.zeros((100, 50))
        freq0 = 1
        freq1 = 3
        tw = 'tw'
        corrs = 'c'
        out = self.dvg._compute_cd(skernels, freq0, freq1, tw, corrs)
        dv_mock.assert_called_once_with(np.array('c'), 2, tw, 2)
        np.testing.assert_array_almost_equal(
            np.eye(100)*dv_mock.return_value**2, out)

    def test_compute_dist(self):
        dist = self.dvg._compute_dist_matrix()
        self.assertTupleEqual(dist.shape, (100, 100))
        # make sure it's taken from self.dist
        dist2 = self.dvg._compute_dist_matrix()
        np.testing.assert_array_equal(dist, dist2)

    @mock.patch('seismic.monitor.spatial.geo2cart')
    def test_find_coord_float(self, g2c_mock: mock.MagicMock):
        g2c_mock.return_value = (self.dvg.xf[3], self.dvg.yf[3])
        ii = self.dvg._find_coord(4, 5)
        g2c_mock.assert_called_once_with(4, 5, self.dvg.lat0, self.dvg.lon0)
        self.assertEqual(ii, 3)

    @mock.patch('seismic.monitor.spatial.geo2cart')
    def test_find_coord_array(self, g2c_mock: mock.MagicMock):
        g2c_mock.return_value = (
            self.dvg.xf[
                np.arange(5, dtype=int)], self.dvg.yf[np.arange(5, dtype=int)])
        ii = self.dvg._find_coord(0, 0)
        np.testing.assert_array_equal(np.arange(5, dtype=int), ii)

    @mock.patch('seismic.monitor.spatial.geo2cart')
    def test_find_coord_float_not_covered(self, g2c_mock: mock.MagicMock):
        g2c_mock.return_value = (100, 500)
        with self.assertRaises(ValueError):
            self.dvg._find_coord(0, 0)

    @mock.patch('seismic.monitor.spatial.geo2cart')
    def test_find_coord_array_not_covered(self, g2c_mock: mock.MagicMock):
        g2c_mock.return_value = (
            np.arange(100),
            np.arange(100) + 500)
        with self.assertRaises(ValueError):
            self.dvg._find_coord(0, 0)

    @mock.patch('seismic.monitor.spatial.geo2cart')
    @mock.patch('seismic.monitor.spatial.sensitivity_kernel')
    def test_compute_sensitivity_kernels(
            self, sk_mock: mock.MagicMock, g2c_mock: mock.MagicMock):
        sk_mock.return_value = np.zeros((2, 2))
        slat0 = 'slat0'
        slon0 = 'slon0'
        slat1 = 'slat1'
        slon1 = 'slon1'
        g2c_mock.side_effect = ([[0], [1]], [[2], [3]])
        out = self.dvg._compute_sensitivity_kernels(
            slat0, slon0, slat1, slon1, 1)
        calls = [
            mock.call(slat0, slon0, self.dvg.lat0, 0),
            mock.call(slat1, slon1, self.dvg.lat0, 0)]
        g2c_mock.assert_has_calls(calls)
        np.testing.assert_array_almost_equal(out, np.array([np.zeros((4))]))

    @mock.patch('seismic.monitor.spatial.geo2cart')
    @mock.patch('seismic.monitor.spatial.sensitivity_kernel')
    def test_compute_sensitivity_kernels_already_computed(
            self, sk_mock: mock.MagicMock, g2c_mock: mock.MagicMock):
        slat0 = '0'
        slon0 = '0'
        slat1 = '1'
        slon1 = '1'
        self.dvg.skernels[(0, 1, 2, 3, 1)] = np.zeros((2, 2))
        g2c_mock.side_effect = ([[0], [1]], [[2], [3]])
        out = self.dvg._compute_sensitivity_kernels(
            slat0, slon0, slat1, slon1, 1)
        calls = [
            mock.call(slat0, slon0, self.dvg.lat0, 0),
            mock.call(slat1, slon1, self.dvg.lat0, 0)]
        g2c_mock.assert_has_calls(calls)
        sk_mock.assert_not_called()
        np.testing.assert_array_almost_equal(out, np.array([np.zeros((2, 2))]))

    @mock.patch('seismic.monitor.spatial.dv_starts')
    @mock.patch('seismic.monitor.spatial.align_dv_curves')
    def test_align_dvs_to_grid_already_aligned(
            self, align_mock: mock.MagicMock, starts_mock: mock.MagicMock):
        dv = mock.MagicMock()
        dv.stats.corr_start = np.arange(100)
        dv.stats.corr_end = dv.stats.corr_start + 1
        dv.value = np.arange(100)
        dv.corr = np.arange(100)/100
        dv.dv_processing = {'aligned': 0}
        out = self.dvg.align_dvs_to_grid([dv], 0, 0, 0)
        self.assertEqual(len(out), 0)
        starts_mock.assert_not_called()
        align_mock.assert_not_called()

    @mock.patch('seismic.monitor.spatial.dv_starts')
    @mock.patch('seismic.monitor.spatial.align_dv_curves')
    def test_align_dvs_to_grid_not_started(
            self, align_mock: mock.MagicMock, starts_mock: mock.MagicMock):
        dv = mock.MagicMock()
        dv.stats.corr_start = np.arange(100)
        dv.stats.corr_end = dv.stats.corr_start + 1
        dv.value = np.arange(100)
        dv.corr = np.arange(100)/100
        dv.dv_processing = {'aligned': False}
        starts_mock.return_value = False
        out = self.dvg.align_dvs_to_grid([dv], 0, 0, 0)
        self.assertEqual(len(out), 0)
        starts_mock.assert_called_once_with(dv, 0, 0)
        align_mock.assert_not_called()

    @mock.patch('seismic.monitor.spatial.dv_starts')
    @mock.patch('seismic.monitor.spatial.align_dv_curves')
    def test_align_dvs_to_grid_no_grid(
            self, align_mock: mock.MagicMock, starts_mock: mock.MagicMock):
        dv = mock.MagicMock()
        dv.stats.corr_start = np.arange(100)
        dv.stats.corr_end = dv.stats.corr_start + 1
        dv.value = np.arange(100)
        dv.corr = np.arange(100)/100
        dv.dv_processing = {'aligned': False}
        starts_mock.return_value = True
        out = self.dvg.align_dvs_to_grid([dv], 0, 0, 0)
        self.assertEqual(len(out), 1)
        starts_mock.assert_called_once_with(dv, 0, 0)
        align_mock.assert_called_once_with(dv, 0, 0, 0)

    @mock.patch('seismic.monitor.spatial.dv_starts')
    @mock.patch('seismic.monitor.spatial.align_dv_curves')
    def test_align_dvs_to_grid_grid1(
            self, align_mock: mock.MagicMock, starts_mock: mock.MagicMock):
        dv = mock.MagicMock()
        dv.stats.corr_start = np.arange(100)
        dv.stats.corr_end = dv.stats.corr_start + 1
        dv.stats.id = 'b'
        dv.value = np.arange(100)
        dv.corr = np.arange(100)/100
        dv.dv_processing = {'aligned': False}
        starts_mock.return_value = True
        dvg = deepcopy(self.dvg)
        dvg.vel_change = 1
        with mock.patch.object(dvg, 'forward_model') as fwd_mock:
            fwd_mock.return_value = [1]
            out = dvg.align_dvs_to_grid([dv], 0, 0, 0, save='a')
            fwd_mock.assert_called_once_with(
                1, [dv], 0)
        dv.save.assert_called_once_with(os.path.join('a', 'DV-b.npz'))
        self.assertEqual(len(out), 1)
        starts_mock.assert_called_once_with(dv, 0, 0)
        align_mock.assert_called_once_with(dv, 0, 0, 1)

    @mock.patch('seismic.monitor.spatial.dv_starts')
    @mock.patch('seismic.monitor.spatial.align_dv_curves')
    def test_align_dvs_to_grid_fwd_nan(
            self, align_mock: mock.MagicMock, starts_mock: mock.MagicMock):
        dv = mock.MagicMock()
        dv.stats.corr_start = np.arange(100)
        dv.stats.corr_end = dv.stats.corr_start + 1
        dv.stats.id = 'b'
        dv.value = np.arange(100)
        dv.corr = np.arange(100)/100
        dv.dv_processing = {'aligned': False}
        starts_mock.return_value = True
        dvg = deepcopy(self.dvg)
        dvg.vel_change = 1
        with mock.patch.object(dvg, 'forward_model') as fwd_mock:
            fwd_mock.return_value = [np.nan]
            with self.assertRaises(ValueError):
                dvg.align_dvs_to_grid([dv], 0, 0, 0)
            fwd_mock.assert_called_once_with(
                1, [dv], 0)
        starts_mock.assert_called_once_with(dv, 0, 0)


class TestSpatial(unittest.TestCase):
    def setUp(self):
        stats = {
            'corr_start': np.array(
                [UTCDateTime(n*3600) for n in np.arange(100)]),
            'id': 'test_object', 'dv_processing': {}}
        self.dv = DV(
            np.ones(100), np.ones(100), value_type='dv', sim_mat=[],
            second_axis=[],
            method='stretch', stats=CorrStats(stats), dv_processing={})
        return super().setUp()

    def test_align_dv_curves_const(self):
        utc = UTCDateTime(0)  # Set a dummy UTCDateTime
        steps = 10  # Set the number of steps
        value = 0.5  # Set the value for alignment
        # Call the align_dv_curves function
        aligned_dv = deepcopy(self.dv)
        spt.align_dv_curves(aligned_dv, utc, steps, value)
        np.testing.assert_array_equal(aligned_dv.value, 0.5)
        self.assertDictEqual(aligned_dv.dv_processing, {'aligned': value})

    def test_align_dv_curves_lin(self):
        utc = UTCDateTime(3600*5)  # Set a dummy UTCDateTime
        steps = 4  # Set the number of steps
        value = 0  # Set the value for alignment
        # Call the align_dv_curves function
        aligned_dv = deepcopy(self.dv)
        aligned_dv.value = np.arange(100, dtype=float)
        spt.align_dv_curves(aligned_dv, utc, steps, value)
        np.testing.assert_array_equal(
            aligned_dv.value, np.arange(100, dtype=float)-5)
        self.assertDictEqual(aligned_dv.dv_processing, {'aligned': value})

    def test_align_dv_curves_lin2(self):
        utc = UTCDateTime(3600*99)  # Set a dummy UTCDateTime
        steps = 4  # Set the number of steps
        value = 0  # Set the value for alignment
        # Call the align_dv_curves function
        aligned_dv = deepcopy(self.dv)
        aligned_dv.value = np.arange(100, dtype=float)
        spt.align_dv_curves(aligned_dv, utc, steps, value)
        np.testing.assert_array_equal(
            aligned_dv.value, np.arange(100, dtype=float)-97)
        self.assertDictEqual(aligned_dv.dv_processing, {'aligned': value})

    def test_align_dv_all_nan(self):
        utc = UTCDateTime(3600*99)  # Set a dummy UTCDateTime
        steps = 4  # Set the number of steps
        value = 0  # Set the value for alignment
        # Call the align_dv_curves function
        aligned_dv = deepcopy(self.dv)
        aligned_dv.value = np.arange(100, dtype=float) + np.nan
        with self.assertRaises(ValueError):
            spt.align_dv_curves(aligned_dv, utc, steps, value)


class TestDvStarts(unittest.TestCase):
    def test_short_dv_object(self):
        dv = DV(
            np.ones(3), np.ones(3), value_type='dv', sim_mat=[],
            second_axis=[],
            method='stretch', stats=CorrStats({'id': 'test_object'}))
        self.assertFalse(spt.dv_starts(dv, 0, 0))

    def test_before_dv_start(self):
        stats = {
            'corr_start': np.array(
                [UTCDateTime(n*3600+5e3) for n in np.arange(100)]),
            'id': 'test_object', 'dv_processing': {}}
        dv = DV(
            np.ones(100), np.ones(100), value_type='dv', sim_mat=[],
            second_axis=[],
            method='stretch', stats=CorrStats(stats))
        self.assertFalse(spt.dv_starts(dv, 0, 0))

    def test_after_dv_start(self):
        stats = {
            'corr_start': np.array(
                [UTCDateTime(n*3600+5e3) for n in np.arange(100)]),
            'id': 'test_object', 'dv_processing': {}}
        stats['corr_end'] = stats['corr_start']+3600
        dv = DV(
            np.ones(100), np.ones(100), value_type='dv', sim_mat=[],
            second_axis=[],
            method='stretch', stats=CorrStats(stats))
        self.assertFalse(spt.dv_starts(dv, UTCDateTime(1e7), 0))

    def test_low_corr(self):
        stats = {
            'corr_start': np.array(
                [UTCDateTime(n*3600) for n in np.arange(100)]),
            'id': 'test_object', 'dv_processing': {}}
        stats['corr_end'] = stats['corr_start']+3600
        dv = DV(
            np.ones(100)*0.1, np.ones(100), value_type='dv', sim_mat=[],
            second_axis=[],
            method='stretch', stats=CorrStats(stats))
        self.assertFalse(spt.dv_starts(dv, UTCDateTime(3600), 0.2))

    def test_true(self):
        stats = {
            'corr_start': np.array(
                [UTCDateTime(n*3600) for n in np.arange(100)]),
            'id': 'test_object', 'dv_processing': {}}
        stats['corr_end'] = stats['corr_start']+3600
        dv = DV(
            np.ones(100)*0.7, np.ones(100), value_type='dv', sim_mat=[],
            second_axis=[],
            method='stretch', stats=CorrStats(stats))
        self.assertTrue(spt.dv_starts(dv, UTCDateTime(3600), 0.6))


if __name__ == "__main__":
    unittest.main()
