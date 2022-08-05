'''
:copyright:

:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 24th June 2021 02:23:40 pm
Last Modified: Friday, 5th August 2022 01:01:19 pm
'''

import unittest

import numpy as np

from seismic.monitor import stretch_mod as sm


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


class TestTimeShiftApply(unittest.TestCase):
    def test_result(self):
        shift = (np.random.random()*10) - 5
        shifta = np.array([shift])
        # number of points for new
        corr = np.arange(100, dtype=np.float64)
        corr_shift = sm.time_shift_apply(corr, shifta, single_sided=True)
        shift = int(np.ceil(np.abs(shift))*np.sign(shift))
        # Zeros on edges
        if shift > 0:
            np.testing.assert_array_equal(corr_shift[0][:shift], 0)
            np.testing.assert_equal(
                np.floor(corr_shift[0][shift:]), corr[:-shift])
        else:
            np.testing.assert_array_equal(corr_shift[0][shift:], 0)
            np.testing.assert_equal(
                np.floor(corr_shift[0][:shift]), corr[-shift-1:-1])

    def test_result_two_sided(self):
        shift = (np.random.random()*10) - 5
        shifta = np.array([shift])
        # number of points for new
        corr = np.arange(-50, 51, dtype=np.float64)
        corr_shift = sm.time_shift_apply(corr, shifta, single_sided=False)
        shift = int(np.ceil(np.abs(shift))*np.sign(shift))

        # Zeros on edges
        if shift > 0:
            np.testing.assert_equal(corr_shift[0][:shift], 0)
            np.testing.assert_equal(
                np.floor(corr_shift[0][shift:]), corr[:-shift])
        else:
            np.assert_equal(corr_shift[0][shift:], 0)
            np.testing.assert_equal(
                np.floor(corr_shift[0][:shift]), corr[-shift-1:-1])

    def test_result_inconstant(self):
        shift = np.atleast_2d(np.concatenate((np.zeros(50), -np.ones(50))))
        # number of points for new
        corr = np.arange(100, dtype=np.float64)
        corr_shift = sm.time_shift_apply(corr, shift, single_sided=True)
        # Zeros on edges
        np.testing.assert_equal(corr_shift[0][:50], corr[:50])
        np.testing.assert_equal(corr_shift[0][50:-1], corr[51:])
        np.testing.assert_equal(corr_shift[0][-1], 0)


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


if __name__ == "__main__":
    unittest.main()
