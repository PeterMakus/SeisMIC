'''
:copyright:

:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 24th June 2021 02:23:40 pm
Last Modified: Tuesday, 8th November 2022 03:56:47 pm
'''

import unittest

import numpy as np

from seismic.monitor import stretch_mod as sm
from seismic.correlate.stats import CorrStats


class TestTimeWindowsCreation(unittest.TestCase):
    def test_wrong_leng(self):
        with self.assertRaises(ValueError):
            sm.time_windows_creation([0, 1, 2], [0, 2])

    def test_negative_step(self):
        win_len = np.random.randint(-100, 0)
        with self.assertRaises(ValueError):
            sm.time_windows_creation([0], win_len)

    def test_one_window(self):
        st = np.random.randint(-10, 10)
        win_len = np.random.randint(2, 100)
        out = sm.time_windows_creation([st], win_len)
        self.assertEqual(win_len, len(out[0]))
        self.assertEqual(out[0][0], st)


# pretty complex to test
# This is a little more like an integral test
class TestTimeStretchEstimate(unittest.TestCase):
    def setUp(self):
        self.n = 1000
        self.ref = np.cos(np.linspace(0, 40*np.pi, self.n, endpoint=True))

    def test_result(self):
        stretch = np.arange(1, 11, 1)/100  # in per cent
        # number of points for new
        nn = ((1+stretch)*self.n)
        corr = np.empty((len(nn), self.n))
        for ii, n in enumerate(nn):
            x = np.linspace(0, 40*np.pi, int(n), endpoint=True)
            jj = int(round(abs(len(x)-self.n)/2))
            corr[ii, :] = np.cos(x)[jj:-jj][:self.n]
        dv = sm.time_stretch_estimate(corr, self.ref, stretch_steps=101)
        self.assertTrue(np.allclose(dv['value'], stretch, atol=0.004))

    def test_no_stretch(self):
        corr = np.tile(self.ref, (4, 1))
        dv = sm.time_stretch_estimate(corr, self.ref, stretch_steps=101)
        self.assertTrue(np.all(dv['value'] == [0, 0, 0, 0]))

    def test_neg_stretch(self):
        stretch = np.arange(1, 11, 1)/100  # in per cent
        # number of points for new
        nn = ((1+stretch)*self.n)
        corr = np.empty((len(nn), self.n))
        # inter = interp1d(self.xref, self.ref, kind='cubic')
        for ii, n in enumerate(nn):
            x = np.linspace(0, 40*np.pi, int(n), endpoint=True)
            jj = int(round(abs(len(x)-self.n)/2))
            corr[ii, :] = np.cos(x)[jj:-jj][:self.n]
        # make the most stretch trace the ref trace
        ref = corr[-1, :]
        dv = sm.time_stretch_estimate(corr[:-1, :], ref, stretch_steps=101)
        self.assertTrue(
            np.allclose(dv['value'], -np.flip(stretch[:-1]), atol=0.004))


class TestTimeShiftEstimate(unittest.TestCase):
    def setUp(self):
        self.n = 1000
        self.ref = np.cos(np.linspace(0, 40*np.pi, self.n, endpoint=True))

    def test_result(self):
        shift = np.arange(0, 10, 1)
        # number of points for new
        corr = np.empty((len(shift), self.n))
        for ii in range(corr.shape[0]):
            corr[ii] = np.roll(self.ref, ii)
        dv = sm.time_shift_estimate(corr, self.ref, shift_steps=21)
        np.testing.assert_array_equal(dv['value'], shift)

    def test_no_stretch(self):
        corr = np.tile(self.ref, (4, 1))
        dv = sm.time_shift_estimate(corr, self.ref, shift_steps=101)
        np.testing.assert_array_equal(dv['value'], [0, 0, 0, 0])

    def test_neg_shift(self):
        shift = -np.arange(0, 10, 1)
        # number of points for new
        corr = np.empty((len(shift), self.n))
        for ii in range(corr.shape[0]):
            corr[ii] = np.roll(self.ref, -ii)
        dv = sm.time_shift_estimate(corr, self.ref, shift_steps=21)
        np.testing.assert_array_equal(dv['value'], shift)


class TestCreateShiftedRefMat(unittest.TestCase):
    def setUp(self):
        self.ref_trc = np.arange(101.)
        self.stats = CorrStats({
            'start_lag': -50,
            'delta': 1,
            'npts': 101})
        self.shifts = np.linspace(-5, 5, 101)

    def test_invalid_shape(self):
        with self.assertRaises(AssertionError):
            sm.create_shifted_ref_mat(np.ones((5, 5)), None, None)

    def test_result(self):
        exp = np.array([self.ref_trc + s for s in self.shifts])
        out = sm.create_shifted_ref_mat(self.ref_trc, self.stats, self.shifts)
        np.testing.assert_array_almost_equal(out, exp)


class TestCompareWithModifiedReference(unittest.TestCase):
    def setUp(self):
        self.ref_trc = np.arange(101.)
        self.stats = CorrStats({
            'start_lag': -50,
            'delta': 1,
            'npts': 101})
        self.shifts = np.linspace(-5, 5, 101)
        self.ref_mat = np.array([self.ref_trc + s for s in self.shifts])
        self.data = np.roll(self.ref_mat, shift=1, axis=0)

    def test_result(self):
        indices = np.arange(101)
        sim_mat = sm.compare_with_modified_reference(
            self.data, self.ref_mat, indices)
        # positions were corr should be 1
        ii = np.hstack((-1, np.arange(0, 100)))
        np.testing.assert_array_almost_equal(sim_mat[np.arange(101), ii], 1)


if __name__ == "__main__":
    unittest.main()
