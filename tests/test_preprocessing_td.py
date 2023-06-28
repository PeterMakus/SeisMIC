'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 03:54:28 pm
Last Modified: Wednesday, 28th June 2023 03:37:06 pm
'''
from copy import deepcopy
import unittest
from unittest import mock

import numpy as np
from scipy.fftpack import next_fast_len
from scipy.signal.windows import gaussian

from seismic.correlate import preprocessing_td as pptd


class TestClip(unittest.TestCase):
    def test_result(self):
        args = {}
        args['std_factor'] = np.random.randint(2, 4)
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1))
        res = pptd.clip(A.copy(), args, {})
        self.assertAlmostEqual(
            np.std(A, axis=1)[0]*args['std_factor'], abs(res).max(axis=1)[0])

    def test_std_0(self):
        args = {}
        args['std_factor'] = np.random.randint(2, 4)
        A = np.ones((5, 100))
        res = pptd.clip(A.copy(), args, {})
        self.assertTrue(np.all(res == np.zeros_like(A)))


class TestDetrend(unittest.TestCase):
    def test_contains_nan(self):
        data = np.ones((25, 50))
        data[1:3, 9:12] = np.nan
        args = {'type': 'constant'}
        out = pptd.detrend(data, args, {})
        np.testing.assert_almost_equal(out[np.logical_not(np.isnan(out))], 0)

    def test_result_linear(self):
        data = np.arange(500)
        args = {'type': 'linear'}
        out = pptd.detrend(data, args, {})
        np.testing.assert_almost_equal(out, 0)

    def test_result_const_2d(self):
        data = np.vstack((np.ones((2, 500)), np.arange(500)))
        exp = np.vstack((np.zeros((2, 500)), np.arange(500)-249.5))
        args = {'type': 'constant'}
        out = pptd.detrend(data, args, {})
        np.testing.assert_almost_equal(out, exp)

    def test_result_linear_2d(self):
        data = np.vstack(
            (np.cos(np.linspace(0, 2*np.pi, 500)), np.arange(500)))
        exp = np.vstack(
            (np.cos(np.linspace(0, 2*np.pi, 500)), np.zeros((500))))
        args = {'type': 'linear'}
        out = pptd.detrend(data, args, {})
        np.testing.assert_almost_equal(out, exp, decimal=2)

    def test_only_nans(self):
        data = np.ones((50, 25))
        data.fill(np.nan)
        args = {'type': 'linear'}
        out = pptd.detrend(data, args, {})
        np.testing.assert_almost_equal(out, data)


class TestMute(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.params['sampling_rate'] = 25

    def test_return_zeros(self):
        # function is supposed to return zeros if input shorter than
        # the taper length
        npts = np.random.randint(100, 599)
        A = np.ones((npts, np.random.randint(2, 55)))
        args = {}
        args['taper_len'] = self.params['sampling_rate']*(
            npts + np.random.randint(1, 99))
        self.assertTrue(
            np.all(pptd.mute(A, args, self.params) == np.zeros_like(A)))

    def test_taper_len_error(self):
        args = {}
        args['taper_len'] = 0
        A = np.ones((5, 2))
        with self.assertRaises(ValueError):
            pptd.mute(A, args, self.params)

    def test_mute_std(self):
        # testing the actual muting of the bit
        args = {}
        args['taper_len'] = 1
        args['extend_gaps'] = True
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1)).T
        res = pptd.mute(A.copy(), args, self.params)
        self.assertLessEqual(
            res[:, 0].max(axis=0), np.std(A))

    def test_mute_std_factor(self):
        # testing the actual muting of the bit
        args = {}
        args['taper_len'] = 1
        args['extend_gaps'] = True
        args['std_factor'] = np.random.randint(1, 5)
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1)).T
        res = pptd.mute(A.copy(), args, self.params)
        self.assertLessEqual(
            res[:, 0].max(axis=0),
            args['std_factor']*np.std(A))

    def test_mute_absolute(self):
        args = {}
        args['taper_len'] = 1
        args['extend_gaps'] = True
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1)).T
        args['threshold'] = A[:, 0].max(axis=0)/np.random.randint(2, 4)
        res = pptd.mute(A.copy(), args, self.params)
        self.assertLessEqual(
            res[:, 0].max(axis=0), args['threshold'])

    @mock.patch('seismic.correlate.preprocessing_td.TDfilter')
    def test_mute_absolute_w_filter(self, tdf_mock):
        args = {}
        args['taper_len'] = 1
        args['extend_gaps'] = False
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1))
        args['threshold'] = A[0].max()/np.random.randint(2, 4)
        args['filter'] = 'blub'
        tdf_mock.return_value = deepcopy(A)
        res = pptd.mute(A, args, self.params)
        self.assertLessEqual(
            res[0].max(), args['threshold'])
        tdf_mock.assert_called_once_with(A, 'blub', self.params)


class TestNormalizeStd(unittest.TestCase):
    def test_result(self):
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1))
        res = pptd.normalizeStandardDeviation(A, {}, {})
        self.assertAlmostEqual(np.std(res, axis=1)[0], 1)

    def test_std_0(self):
        # Feed in DC signal to check this
        A = np.ones((250, 2))
        res = pptd.normalizeStandardDeviation(A.copy(), {}, {})
        self.assertTrue(np.all(res == A))


class TestFDSignBitNormalisation(unittest.TestCase):
    # Not much to test here
    def test_result(self):
        np.random.seed(2)
        dim = (np.random.randint(200, 766), np.random.randint(2, 44))
        A = np.random.random(dim)-.5
        expected_result = np.sign(A)
        self.assertTrue(np.allclose(
            expected_result, pptd.signBitNormalization(A, {}, {})))


class TestTaper(unittest.TestCase):
    def test_cosine_filter(self):
        data = np.ones((2, 50))
        exp = deepcopy(data)
        exp[:, 0] = 0
        exp[:, -1] = 0
        exp[:, 1] = 0.5
        exp[:, -2] = .5
        args = {'type': 'cosine_taper', 'p': 0.1}
        out = pptd.taper(data, args, {})
        np.testing.assert_allclose(out, exp)

    def test_other_taper(self):
        data = np.ones((2, 50))
        args = {'type': 'hann'}
        out = pptd.taper(deepcopy(data), args, {})
        np.testing.assert_array_less(out, data)


class TestTDNormalisation(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.params['sampling_rate'] = 25

    def test_win_length_error(self):
        args = {}
        args['windowLength'] = 0
        with self.assertRaises(ValueError):
            pptd.TDnormalization(np.ones((5, 2)), args, self.params)

    def test_result_no_smooth(self):
        args = {}
        args['windowLength'] = 1/25
        args['filter'] = False
        A = np.random.random((25, 25))
        res = pptd.TDnormalization(A, args, self.params)
        # instabilities cause the envelope to become higher
        np.testing.assert_array_less(res, 1)
        np.testing.assert_allclose(res, 1, atol=1)

    @mock.patch('seismic.correlate.preprocessing_td.TDfilter')
    def test_result_no_smooth_filt(self, tdf_mock):
        args = {}
        args['windowLength'] = 1/25
        args['filter'] = 'blub'
        A = np.random.random((25, 25))
        tdf_mock.return_value = deepcopy(A)
        res = pptd.TDnormalization(A, args, self.params)
        # instabilities cause the envelope to become higher
        np.testing.assert_array_less(res, 1)
        np.testing.assert_allclose(res, 1, atol=1)
        tdf_mock.assert_called_once_with(A, 'blub', self.params)


class TestTDFilter(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.params['sampling_rate'] = 100

    def test_bandpass(self):
        args = {'type': 'bandpass', 'freqmin': 0.01, 'freqmax': 0.02}
        A = np.sin(np.linspace(0, 10*np.pi, 50)).T
        out = pptd.TDfilter(A, args, self.params)
        np.testing.assert_allclose(out, 0, atol=1e-8)


class TestZeroPadding(unittest.TestCase):
    def setUp(self):
        self.params = {'sampling_rate': 25, 'lengthToSave': 200}
        self.A = np.empty(
            (np.random.randint(100, 666), np.random.randint(2, 45)))

    def test_result_next_fast_len(self):
        expected_len = next_fast_len(self.A.shape[1])
        self.assertEqual(pptd.zeroPadding(
            self.A, {'type': 'nextFastLen'}, self.params).shape[1],
            expected_len)

    def test_result_avoid_wrap_around(self):
        expected_len = self.A.shape[1] + \
            self.params['sampling_rate'] * self.params['lengthToSave']
        self.assertEqual(pptd.zeroPadding(
            self.A, {'type': 'avoidWrapAround'}, self.params).shape[1],
            expected_len)

    def test_result_avoid_wrap_fast_len(self):
        expected_len = next_fast_len(int(
            self.A.shape[1]
            + self.params['sampling_rate'] * self.params['lengthToSave']))
        self.assertEqual(pptd.zeroPadding(
            self.A, {'type': 'avoidWrapFastLen'}, self.params).shape[1],
            expected_len)

    def test_result_next_fast_len_axis1(self):
        expected_len = next_fast_len(self.A.shape[1])
        self.assertEqual(pptd.zeroPadding(
            self.A, {'type': 'nextFastLen'}, self.params, axis=1).shape[1],
            expected_len)

    def test_result_avoid_wrap_around_axis1(self):
        expected_len = self.A.shape[1]\
            + self.params['sampling_rate'] * self.params['lengthToSave']
        self.assertEqual(pptd.zeroPadding(
            self.A, {'type': 'avoidWrapAround'}, self.params, axis=1).shape[1],
            expected_len)

    def test_result_avoid_wrap_fast_len_axis1(self):
        expected_len = next_fast_len(int(
            self.A.shape[1]
            + self.params['sampling_rate'] * self.params['lengthToSave']))
        self.assertEqual(pptd.zeroPadding(
            self.A,
            {'type': 'avoidWrapFastLen'}, self.params, axis=1).shape[1],
            expected_len)

    def test_weird_axis(self):
        with self.assertRaises(NotImplementedError):
            pptd.zeroPadding(self.A, {}, {}, axis=7)

    def test_higher_dim(self):
        with self.assertRaises(NotImplementedError):
            pptd.zeroPadding(np.ones((3, 3, 3)), {}, {})

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            pptd.zeroPadding(self.A, {'type': 'blub'}, self.params)

    def test_empty_array(self):
        B = np.array([])
        with self.assertRaises(ValueError):
            pptd.zeroPadding(B, {'type': 'nextFastLen'}, self.params)


if __name__ == "__main__":
    unittest.main()
