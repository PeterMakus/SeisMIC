'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 25th June 2021 09:33:09 am
Last Modified: Friday, 25th June 2021 12:32:49 pm
'''

import unittest

import numpy as np
from obspy import UTCDateTime

from miic3.monitor import post_corr_process as pcp


class TestSmooth(unittest.TestCase):
    def test_wrong_dim(self):
        with self.assertRaises(ValueError):
            pcp._smooth(np.empty((5, 5)), window_len=4)

    def test_win_larger_than_array(self):
        win_size = np.random.randint(5, 500)
        win = np.ones((win_size,))
        win_len = win_size + np.random.randint(1, 100)
        with self.assertRaises(ValueError):
            pcp._smooth(win, win_len)

    def test_win_len_sm3(self):
        win_size = np.random.randint(5, 500)
        win = np.random.rand(win_size)
        out = pcp._smooth(win, 2)
        self.assertTrue(np.all(out == win))

    def test_unknown_win(self):
        with self.assertRaises(ValueError):
            pcp._smooth(np.empty((5,)), window_len=4, window='blabla')

    def test_result_mv_av(self):
        win = np.random.rand(np.random.randint(20, 30))
        win_len = (int(len(win)/2) - np.random.randint(3, 7))*2
        out = pcp._smooth(win, win_len, 'flat')
        self.assertAlmostEqual(
            np.average(win[:win_len]), out[int(round(win_len/2))])

    def test_result(self):
        win = np.sin(np.linspace(0, 2*np.pi, endpoint=True))
        out = pcp._smooth(win, 20)
        self.assertLess(out.max(), win.max())


class TestCorrMatResample(unittest.TestCase):
    def test_wrong_len_end_times(self):
        with self.assertRaises(ValueError):
            pcp.corr_mat_resample([], {}, [0]*25, [1, 2])

    def test_result(self):
        corr_data = np.empty((11, 500))
        starts = [UTCDateTime(ii) for ii in np.arange(0, 101, 10)]
        stats = {'corr_start': starts}
        nstarts = [UTCDateTime(ii) for ii in np.arange(0, 101, 20)]
        outdata, outstats = pcp.corr_mat_resample(corr_data, stats, nstarts)
        self.assertEqual(outdata.shape[0], 6)
        self.assertTrue(np.all(outstats['corr_start'] == nstarts))
        self.assertTrue(np.allclose(np.mean(corr_data[:1, :]), outdata[0]))

    def test_gaps(self):
        corr_data = np.empty((10, 500))
        starts = np.hstack((np.arange(0, 31, 10), np.arange(50, 101, 10)))
        starts = [UTCDateTime(ii) for ii in starts]
        ends = [ii + 10 for ii in starts]
        stats = {'corr_start': starts, 'corr_end': ends}
        nstarts = [UTCDateTime(ii) for ii in np.arange(0, 101, 10)]
        outdata, outstats = pcp.corr_mat_resample(corr_data, stats, nstarts)
        self.assertEqual(outdata.shape[0], 11)
        self.assertTrue(np.all(np.isnan(outdata[4, :])))





if __name__ == "__main__":
    unittest.main()
