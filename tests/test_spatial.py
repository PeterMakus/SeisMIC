'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 16th January 2023 11:07:27 am
Last Modified: Tuesday, 17th January 2023 11:57:13 am
'''

import unittest
from unittest import mock
import warnings

import numpy as np

from seismic.monitor import spatial as spt


class TestProbability(unittest.TestCase):
    def test_no_chance(self):
        # distance larger than t*v
        dist = np.random.rand() + 3
        t = np.random.rand() + 5
        c = dist/t - np.random.rand()
        # mf path should not matter
        self.assertEqual(spt.probability(dist, t, c, 10), 0)

    def test_vel0(self):
        with self.assertRaises(ZeroDivisionError):
            spt.probability(1, 1, 0, 10), 1

    def test_prob1(self):
        # distance=t*v
        dist = 3
        t = 5
        c = dist/t
        # mf path should not matter
        self.assertEqual(spt.probability(dist, t, c, 10), 1)

    def test_prob1_array(self):
        # distance=t*v
        dist = np.ones(40)*2
        t = 5
        c = dist/t
        # mf path should not matter
        np.testing.assert_almost_equal(spt.probability(dist, t, c, 10), 1)

    def test_prob_rand(self):
        # in any other case this should be 0 < probability < 1
        dist = np.random.rand() + 3
        t = np.random.rand() + 5
        c = dist/t + .01 + np.random.rand()
        # mf path should not matter
        self.assertGreater(spt.probability(dist, t, c, 10), 0)
        self.assertLess(spt.probability(dist, t, c, 10), 1)

    def test_dist_neg(self):
        dist = -3
        t = 5
        c = dist/t
        with self.assertRaises(ValueError):
            spt.probability(dist, t, c, 10)


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
        dt = .1
        s1 = s2 = np.array([5, 5])
        x = y = np.arange(10)
        t = 20
        vel = mf_path = 1
        cgd_mock.return_value = np.zeros((10, 10))
        pbb_mock.return_value = np.ones((10, 10))
        K = spt.sensitivity_kernel(s1, s2, x, y, t, dt, vel, mf_path)
        cgd_calls = [
            mock.call(x, y, s1[0], s1[1]),
            mock.call(x, y, s2[0], s2[1])
        ]
        cgd_mock.assert_has_calls(cgd_calls)
        # To be continued here
        self.assertEqual(pbb_mock.call_count, 403)

        calls = [mock.call(0, 20, 1, 1, atol=dt/2)]
        for tt in np.arange(0, t + dt, dt):
            calls.extend([
                mock.call(mock.ANY, tt, vel, mf_path, atol=dt/2),
                mock.call(mock.ANY, t-tt, vel, mf_path, atol=dt/2)])
        pbb_mock.assert_has_calls(calls)

        np.testing.assert_almost_equal(K, np.ones((10, 10))*t)

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

    def test_raises_error_for_neg_corr(self):
        corr = -1
        bw = 2
        tw = (5, 10)
        freq_c = 3
        with self.assertRaises(ValueError):
            spt.data_variance(corr, bw, tw, freq_c)


if __name__ == "__main__":
    unittest.main()
