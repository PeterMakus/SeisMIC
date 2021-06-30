'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 25th June 2021 09:33:09 am
Last Modified: Wednesday, 30th June 2021 05:12:13 pm
'''

import unittest

import numpy as np
from obspy import UTCDateTime
from obspy.core.trace import Stats

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


class TestCorrMatFilter(unittest.TestCase):
    def setUp(self):
        self.data = np.tile([
            np.cos(np.linspace(0, 4*np.pi, 512)) +  # .0625 Hz
            np.cos(np.linspace(0, 400*np.pi, 512)) +  # 6.25 Hz
            np.cos(np.linspace(0, 200*np.pi, 512))], (2, 1))  # 3.125 Hz
        self.dataft = abs(np.fft.rfft(self.data, axis=1))
        self.f = np.fft.rfftfreq(512, 1/16)
        self.stats = Stats({'sampling_rate': 16})

    def test_result(self):
        datafilt = pcp.corr_mat_filter(self.data.copy(), self.stats, (2, 4))
        datafiltft = abs(np.fft.rfft(datafilt))
        self.assertTrue(np.all(datafilt != self.data))
        lp = np.where(self.f == 0.0625)
        hp = np.where(self.f == 6.25)
        unch = np.where(self.f == 3.125)
        self.assertTrue(
            np.all(datafiltft[:, lp] < self.dataft[:, lp]))
        self.assertTrue(
            np.all(datafiltft[:, hp] < self.dataft[:, hp]))
        self.assertTrue(
            np.allclose(datafiltft[:, unch], self.dataft[:, unch], rtol=0.02))

    def test_wrong_len_freq(self):
        with self.assertRaises(ValueError):
            _ = pcp.corr_mat_filter(self.data.copy(), self.stats, (2, 4, 3))

    def test_lp_0(self):
        with self.assertRaises(ValueError):
            _ = pcp.corr_mat_filter(self.data.copy(), self.stats, (0, 8))

    def test_no_filt(self):
        datafilt = pcp.corr_mat_filter(
            self.data.copy(), self.stats, (0.001, 7.9))
        lp = np.where(self.f == 0.0625)
        datafiltft = abs(np.fft.rfft(datafilt))
        hp = np.where(self.f == 6.25)
        unch = np.where(self.f == 3.125)
        self.assertTrue(
            np.allclose(datafiltft[:, lp], self.dataft[:, lp], rtol=0.001))
        self.assertTrue(
            np.allclose(datafiltft[:, hp], self.dataft[:, hp], rtol=0.005))
        self.assertTrue(
            np.allclose(datafiltft[:, unch], self.dataft[:, unch], rtol=0.001))


class TestCorrMatTrim(unittest.TestCase):
    def setUp(self):
        npts = 501
        self.data = np.tile(
            np.sin(np.linspace(0, np.pi, npts, endpoint=True)), (2, 1))
        self.stats = Stats({
            'npts': npts, 'start_lag': -125, 'end_lag': 125,
            'sampling_rate': 2})

    def test_result(self):
        start = np.random.randint(-124, -1)
        end = np.random.randint(1, 124)
        npts_trim = (end-start)*2+1
        ndata, nstats = pcp.corr_mat_trim(self.data, self.stats, start, end)
        self.assertEqual(nstats.start_lag, start)
        self.assertEqual(nstats.end_lag, end)
        self.assertEqual(nstats.npts, npts_trim)
        self.assertEqual(ndata.shape, (2, npts_trim))

    def test_before(self):
        start = -150
        end = 100
        ndata, _ = pcp.corr_mat_trim(self.data, self.stats, start, end)
        self.assertTrue(np.all(ndata == self.data))

    def test_after(self):
        start = -100
        end = 150
        ndata, _ = pcp.corr_mat_trim(self.data, self.stats, start, end)
        self.assertTrue(np.all(ndata == self.data))

    def test_identical(self):
        start = -125
        end = 125
        ndata, _ = pcp.corr_mat_trim(self.data, self.stats, start, end)
        self.assertTrue(np.all(ndata == self.data))


class TestCorrMatResample(unittest.TestCase):
    def test_wrong_len_end_times(self):
        with self.assertRaises(ValueError):
            pcp.corr_mat_resample([], {}, [0]*25, [1, 2])

    def test_result(self):
        corr_data = np.random.rand(11, 500)
        starts = [UTCDateTime(ii) for ii in np.arange(0, 101, 10)]
        stats = {'corr_start': starts}
        nstarts = [UTCDateTime(ii) for ii in np.arange(0, 101, 20)]
        outdata, outstats = pcp.corr_mat_resample(
            corr_data.copy(), stats, nstarts)
        self.assertEqual(outdata.shape[0], 6)
        self.assertTrue(np.all(outstats['corr_start'] == nstarts))
        self.assertTrue(
            np.allclose(
                np.mean(corr_data[(0, 1), :], axis=0), outdata[0, ]))

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
        self.assertTrue(np.all(outstats['corr_start'] == nstarts))


class CorrMatCorrectDecay(unittest.TestCase):
    def setUp(self):
        # acausal part
        datal = np.exp(np.arange(0, 51, 1))
        self.data = np.tile(
            np.hstack((datal, np.flip(datal[0:-1]))), (2, 1))
        self.stats = Stats({
            'sampling_rate': 1, 'npts': 101, 'start_lag': -50, 'end_lag': 50})

    def test_result(self):
        corrected = pcp.corr_mat_correct_decay(
            self.data.copy(), self.stats.copy())
        # Check that signal is still symmetric
        self.assertTrue(np.all(corrected[:50] == np.flip(corrected[-50:])))
    # ask CSS how this actually works
        

class TestCorrMatNormalize(unittest.TestCase):
    # Don't feel like it's necessary or even helpful to test the result here
    # as I would just be tracking the actual code
    def test_same(self):
        data = np.ones((2, 20))
        stats = {}
        normtypes = ['energy', 'abssum', 'max', 'absmax']
        for n in normtypes:
            norm = pcp.corr_mat_normalize(data.copy(), stats, normtype=n)
            self.assertTrue(np.all(norm == data))

    def test_zero(self):
        data = np.zeros((2, 20))
        stats = {}
        normtypes = ['energy', 'abssum', 'max', 'absmax']
        for n in normtypes:
            norm = pcp.corr_mat_normalize(data.copy(), stats, normtype=n)
            self.assertTrue(np.all(norm == data))

    def test_unknown_type(self):
        data = np.zeros((2, 20))
        stats = {}
        with self.assertRaises(ValueError):
            _ = pcp.corr_mat_normalize(data, stats, normtype='bla')

    def test_start_time(self):
        data = np.hstack((np.ones((2, 10)), 5*np.ones((2, 11))))
        stats = Stats({
            'sampling_rate': 1, 'start_lag': -10, 'end_lag': 10, 'npts': 21})
        normtypes = ['energy', 'abssum', 'max', 'absmax']
        for n in normtypes:
            norm = pcp.corr_mat_normalize(
                data.copy(), stats, starttime=0, normtype=n)
            self.assertTrue(np.all(norm == data/5))

    def test_end_time(self):
        data = np.hstack((5*np.ones((2, 11)), np.ones((2, 10))))
        stats = Stats({
            'sampling_rate': 1, 'start_lag': -10, 'end_lag': 10, 'npts': 21})
        normtypes = ['energy', 'abssum', 'max', 'absmax']
        for n in normtypes:
            norm = pcp.corr_mat_normalize(
                data.copy(), stats, endtime=0, normtype=n)
            self.assertTrue(np.all(norm == data/5))


class TestCorrMatMirror(unittest.TestCase):
    def test_symmetric(self):
        stats = Stats({
            'start_lag': -25, 'end_lag': 25, 'sampling_rate': 10, 'npts': 501})
        data = np.random.rand(2, 501)
        mdata, mstats = pcp.corr_mat_mirror(data.copy(), stats.copy())
        self.assertEqual(mstats['start_lag'], 0)
        self.assertEqual(mstats['end_lag'], stats['end_lag'])
        self.assertEqual(mstats['npts'], 251)
        exp_res = data[:, -251:]
        exp_res[:, 1:] += np.fliplr(data[:, 0:250])
        exp_res[:, 1:] /= 2
        self.assertTrue(np.allclose(exp_res, mdata))

    def test_right_side(self):
        stats = Stats({
            'start_lag': 0, 'end_lag': 25, 'sampling_rate': 10, 'npts': 251})
        data = np.empty((2, 251))
        mdata, mstats = pcp.corr_mat_mirror(data.copy(), stats.copy())
        self.assertEqual(mstats, stats)
        self.assertTrue(np.all(mdata == data))

    def test_left_side(self):
        stats = Stats({
            'start_lag': -25, 'end_lag': 0, 'sampling_rate': 10, 'npts': 251})
        data = np.empty((2, 251))
        mdata, mstats = pcp.corr_mat_mirror(data.copy(), stats.copy())
        self.assertEqual(mstats, stats)
        self.assertTrue(np.all(mdata == data))

    def test_asymmetric_left(self):
        stats = Stats({
            'start_lag': -10, 'end_lag': 25, 'sampling_rate': 10, 'npts': 351})
        data = np.random.rand(2, 351)
        mdata, mstats = pcp.corr_mat_mirror(data.copy(), stats.copy())
        self.assertEqual(mstats['start_lag'], 0)
        self.assertEqual(mstats['end_lag'], stats['end_lag'])
        self.assertEqual(mstats['npts'], 251)
        exp_res = data[:, -251:]
        exp_res[:, 1:101] += np.fliplr(data[:, 0:100])
        exp_res[:, 1:101] /= 2
        self.assertTrue(np.allclose(exp_res, mdata))

    def test_asymmetric_right(self):
        stats = Stats({
            'start_lag': -25, 'end_lag': 10, 'sampling_rate': 10, 'npts': 351})
        data = np.random.rand(2, 351)
        mdata, mstats = pcp.corr_mat_mirror(data.copy(), stats.copy())
        self.assertEqual(mstats['start_lag'], 0)
        self.assertEqual(mstats['end_lag'], 25)
        self.assertEqual(mstats['npts'], 251)
        exp_res = np.zeros((2, 251))
        exp_res[:, :101] = data[:, -101:]
        exp_res[:, 1:251] += np.fliplr(data[:, 0:250])
        exp_res[:, 1:101] /= 2
        self.assertTrue(np.allclose(exp_res, mdata))


class TestCorrMatTaper(unittest.TestCase):
    def setUp(self):
        self.stats = Stats({'npts': 101, 'sampling_rate': 10})
        self.data = np.ones((2, 101))

    def test_result(self):
        tap = pcp.corr_mat_taper(self.data.copy(), self.stats, 5)
        self.assertTrue(np.all(tap[:, :49] < self.data[:, :49]))
        self.assertTrue(np.all(tap[:, -49:] < self.data[:, -49:]))
        self.assertTrue(np.all(self.data[:, 51] == tap[:, 51]))

    def test_taper_len0(self):
        tap = pcp.corr_mat_taper(self.data.copy(), self.stats, 0)
        self.assertTrue(np.all(tap == self.data))

    def test_neg_taper_len(self):
        with self.assertRaises(ValueError):
            _ = pcp.corr_mat_taper(self.data.copy(), self.stats, -5)

    def test_tap_too_long(self):
        with self.assertRaises(ValueError):
            _ = pcp.corr_mat_taper(self.data.copy(), self.stats, 20)


class TestCorrMatTaperCenter(unittest.TestCase):
    def setUp(self):
        self.stats = Stats({
            'npts': 101, 'sampling_rate': 10, 'start_lag': -5, 'end_lag': 5})
        self.data = np.ones((2, 101))

    def test_result(self):
        tap = pcp.corr_mat_taper_center(self.data.copy(), self.stats, 10, 1)
        self.assertTrue(np.all(tap[:, 1:-1] < self.data[:, 1:-1]))
        self.assertTrue(np.all(self.data[:, -1] == tap[:, -1]))
        self.assertTrue(np.all(self.data[:, 0] == tap[:, 0]))

    def test_taper_len0(self):
        tap = pcp.corr_mat_taper_center(self.data.copy(), self.stats, 0)
        self.assertTrue(np.all(tap == self.data))

    def test_neg_taper_len(self):
        with self.assertRaises(ValueError):
            _ = pcp.corr_mat_taper_center(self.data.copy(), self.stats, -5)

    def test_tap_too_long(self):
        with self.assertRaises(ValueError):
            _ = pcp.corr_mat_taper_center(self.data.copy(), self.stats, 20)


if __name__ == "__main__":
    unittest.main()
