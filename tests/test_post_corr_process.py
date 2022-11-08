'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 25th June 2021 09:33:09 am
Last Modified: Tuesday, 8th November 2022 03:26:47 pm
'''

import unittest
from unittest import mock
from copy import deepcopy

import numpy as np
from obspy import UTCDateTime

from seismic.monitor import post_corr_process as pcp
from seismic.correlate.stats import CorrStats
from seismic.monitor.dv import DV


class TestClip(unittest.TestCase):
    def test_axisNone(self):
        A = np.random.random((5, 5))*2 - 1
        stdA = np.std(A)
        A = pcp.corr_mat_clip(A, 1, None)
        self.assertLessEqual(A.max(), stdA)
        self.assertGreaterEqual(A.min(), -stdA)

    def test_axis1(self):
        A = np.random.random((5, 5))*2 - 1
        stdA = np.std(A, axis=1)
        A = pcp.corr_mat_clip(A, .5, 1)
        np.testing.assert_array_less(A.max(axis=1), .5*stdA+1e-4)
        np.testing.assert_array_less(abs(A.min(axis=1)), .5*stdA+1e-4)

    def test_axis0(self):
        A = np.random.random((5, 5))*2 - 1
        stdA = np.std(A, axis=0)
        A = pcp.corr_mat_clip(A, .5, 0)
        np.testing.assert_array_less(A.max(axis=0), .5*stdA+1e-4)
        np.testing.assert_array_less(abs(A.min(axis=0)), .5*stdA+1e-4)


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
            np.cos(np.linspace(0, 4*np.pi, 512))
            + np.cos(np.linspace(0, 400*np.pi, 512))
            + np.cos(np.linspace(0, 200*np.pi, 512))], (2, 1))
        self.dataft = abs(np.fft.rfft(self.data, axis=1))
        self.f = np.fft.rfftfreq(512, 1/16)
        self.stats = CorrStats({'sampling_rate': 16})

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
        self.stats = CorrStats({
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


# class CorrMatCorrectDecay(unittest.TestCase):
#     def setUp(self):
#         # acausal part
#         datal = np.exp(np.linspace(-5, 0, 251, endpoint=True))
#         self.data = np.tile(
#             np.hstack(
#                 (datal,
#                     np.flip(datal[0:-1])))*np.cos(
#                         np.linspace(0, 40*np.pi, 501)),
#                 (2, 1)
#                 )
#         self.stats = CorrStats({
#         'sampling_rate': 1, 'npts': 501, 'start_lag': -50, 'end_lag': 50})

#     def test_result(self):
#         corrected = pcp.corr_mat_correct_decay(
#             self.data.copy(), self.stats.copy())
#         # Check that signal is still symmetric
#         # print(corrected[0])
#         # print(self.data[0])
#         # print(np.sin(np.linspace(0, 40*np.pi, 501)))
#         # This undercorrects
#         # Should maybe be adapted, we leave it failing for now as a reminder
#         self.assertTrue(np.allclose(
#             corrected, np.cos(np.linspace(0, 40*np.pi, 501)), atol=0.1))
#     # ask CSS how this actually works


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
        stats = CorrStats({
            'sampling_rate': 1, 'start_lag': -10, 'end_lag': 10, 'npts': 21})
        normtypes = ['energy', 'abssum', 'max', 'absmax']
        for n in normtypes:
            norm = pcp.corr_mat_normalize(
                data.copy(), stats, starttime=0, normtype=n)
            self.assertTrue(np.all(norm == data/5))

    def test_end_time(self):
        data = np.hstack((5*np.ones((2, 11)), np.ones((2, 10))))
        stats = CorrStats({
            'sampling_rate': 1, 'start_lag': -10, 'end_lag': 10, 'npts': 21})
        normtypes = ['energy', 'abssum', 'max', 'absmax']
        for n in normtypes:
            norm = pcp.corr_mat_normalize(
                data.copy(), stats, endtime=0, normtype=n)
            self.assertTrue(np.all(norm == data/5))


class TestCorrMatMirror(unittest.TestCase):
    def test_symmetric(self):
        stats = CorrStats({
            'start_lag': -25, 'sampling_rate': 10, 'npts': 501})
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
        stats = CorrStats({
            'start_lag': 0, 'sampling_rate': 10, 'npts': 251})
        data = np.empty((2, 251))
        mdata, mstats = pcp.corr_mat_mirror(data.copy(), stats.copy())
        self.assertEqual(mstats, stats)
        self.assertTrue(np.all(mdata == data))

    def test_left_side(self):
        stats = CorrStats({
            'start_lag': -25, 'sampling_rate': 10, 'npts': 251})
        data = np.empty((2, 251))
        mdata, mstats = pcp.corr_mat_mirror(data.copy(), stats.copy())
        self.assertEqual(mstats, stats)
        self.assertTrue(np.all(mdata == data))

    def test_asymmetric_left(self):
        stats = CorrStats({
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
        stats = CorrStats({
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
        self.stats = CorrStats({'npts': 101, 'sampling_rate': 10})
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
        self.stats = CorrStats({
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


class CorrMatResampleTime(unittest.TestCase):
    def setUp(self):
        self.stats = CorrStats({
            'npts': 201, 'sampling_rate': 10, 'start_lag': -10, 'end_lag': 10})

    def test_resample(self):
        data = np.ones((2, 201))
        datars, statsrs = pcp.corr_mat_resample_time(
            data.copy(), self.stats.copy(), 5)
        self.assertEqual(5, statsrs['sampling_rate'])
        self.assertEqual(self.stats['start_lag'], statsrs['start_lag'])
        self.assertEqual(100, statsrs['npts'])
        self.assertEqual(100, datars.shape[1])

    def test_aafilter(self):
        data = np.tile(np.sin(
            np.linspace(0, 150*np.pi, 201, endpoint=True)), (2, 1))
        datars, statsrs = pcp.corr_mat_resample_time(
            data.copy(), self.stats.copy(), 2.5)
        self.assertTrue(np.allclose(datars, np.zeros(datars.shape), atol=0.03))
        self.assertEqual(2.5, statsrs['sampling_rate'])
        self.assertEqual(self.stats['start_lag'], statsrs['start_lag'])
        self.assertEqual(50, statsrs['npts'])
        self.assertEqual(50, datars.shape[1])

    def test_f_higher(self):
        data = np.ones((2, 201))
        with self.assertRaises(ValueError):
            _ = pcp.corr_mat_resample_time(data.copy(), self.stats.copy(), 20)

    def test_f_identical(self):
        data = np.ones((2, 201))
        datars, statsrs = pcp.corr_mat_resample_time(
            data.copy(), self.stats.copy(), 10)
        self.assertTrue(np.all(data == datars))
        self.assertEqual(statsrs, self.stats)


class TestCorrMatDecimate(unittest.TestCase):
    def setUp(self):
        self.stats = CorrStats({
            'npts': 201, 'sampling_rate': 10, 'start_lag': -10, 'end_lag': 10})

    def test_resample(self):
        data = np.ones((2, 201))
        datars, statsrs = pcp.corr_mat_decimate(
            data.copy(), self.stats.copy(), 2)
        self.assertEqual(5, statsrs['sampling_rate'])
        self.assertEqual(self.stats['start_lag'], statsrs['start_lag'])
        self.assertEqual(100, statsrs['npts'])
        self.assertEqual(100, datars.shape[1])

    def test_aafilter(self):
        data = np.tile(np.sin(
            np.linspace(0, 150*np.pi, 201, endpoint=True)), (2, 1))
        datars, statsrs = pcp.corr_mat_decimate(
            data.copy(), self.stats.copy(), 4)
        self.assertTrue(np.allclose(datars, np.zeros(datars.shape), atol=0.06))
        self.assertEqual(2.5, statsrs['sampling_rate'])
        self.assertEqual(self.stats['start_lag'], statsrs['start_lag'])
        self.assertEqual(50, statsrs['npts'])
        self.assertEqual(50, datars.shape[1])

    def test_f_higher(self):
        data = np.ones((2, 201))
        with self.assertRaises(ValueError):
            _ = pcp.corr_mat_decimate(data.copy(), self.stats.copy(), .5)

    def test_f_identical(self):
        data = np.ones((2, 201))
        datars, statsrs = pcp.corr_mat_decimate(
            data.copy(), self.stats.copy(), 1)
        self.assertTrue(np.all(data == datars))
        self.assertEqual(statsrs, self.stats)


class TestCorrMatExtractTrace(unittest.TestCase):
    def setUp(self):
        self.data = np.tile(np.arange(1, 10.1, step=1), (21, 1)).T
        self.stats = CorrStats({
            'start_lag': -10, 'sampling_rate': 1, 'npts': self.data.shape[1]})

    def test_mean(self):
        out = pcp.corr_mat_extract_trace(self.data, self.stats, method='mean')
        exp = np.mean(self.data, axis=0)
        self.assertTrue(np.allclose(out, exp))

    def test_norm_mean(self):
        out = pcp.corr_mat_extract_trace(
            self.data, self.stats, 'norm_mean')
        exp = np.ones(21)
        self.assertTrue(np.allclose(out, exp))

    def test_sim_perc0(self):
        out = pcp.corr_mat_extract_trace(
            self.data, self.stats, 'norm_mean')
        exp = np.ones(21)
        self.assertTrue(np.allclose(out, exp))

    # def test_sim_perc1(self):
    #     # Let's chevck back at some point
    #     indata = np.ones_like(self.data)
    #     randi = np.random.rand(1, 21)
    #     indata += (
    #         np.tile(randi, (10, 1)).T *
    #         np.arange(1, 10.1)).T
    #     out = pcp.corr_mat_extract_trace(
    #         indata, self.stats, 'norm_mean')
    #     print(out)
    #     print(randi)
    #     exp = np.ones(21)
    #     self.assertTrue(np.allclose(out, exp))


class TestCorrMatStretch(unittest.TestCase):
    def setUp(self):
        reftr = np.zeros(101)
        reftr[np.array([25, -25])] = 1
        self.reftr = reftr
        self.stats = CorrStats({
            'start_lag': -50,
            'sampling_rate': 1,
            'corr_start': [UTCDateTime(ii) for ii in range(2)],
            'corr_end': [UTCDateTime(10+ii) for ii in range(2)]
        })

    def test_stretch_0(self):
        data = np.tile(self.reftr, (2, 1))
        dv = pcp.corr_mat_stretch(
            data, self.stats, self.reftr, stretch_steps=101)
        self.assertListEqual([0, 0], list(dv['value']))

    def test_stretch(self):
        s = np.zeros_like(self.reftr)
        s[np.array([26, -26])] = 1
        data = np.vstack((self.reftr, s))

        dv = pcp.corr_mat_stretch(
            data, self.stats, self.reftr, stretch_steps=101)
        self.assertTrue(np.allclose(
            [0, -0.04], list(dv['value'])))

    def test_stretch_single(self):
        s = np.zeros_like(self.reftr)
        s[np.array([26])] = 1
        data = np.vstack((self.reftr, s))

        dv = pcp.corr_mat_stretch(
            data, self.stats, self.reftr, stretch_steps=101, sides='single',
            tw=[np.arange(51)])
        # Flips the result as it assumes 0 lag to be on index 0
        self.assertTrue(np.allclose(
            [0, 0.04], list(dv['value'])))


class TestCorrMatShift(unittest.TestCase):
    def setUp(self):
        reftr = np.zeros(101)
        reftr[np.array([25, -25])] = 1
        self.reftr = reftr
        self.stats = CorrStats({
            'start_lag': -50,
            'sampling_rate': 1,
            'corr_start': [UTCDateTime(ii) for ii in range(2)],
            'corr_end': [UTCDateTime(10+ii) for ii in range(2)]
        })

    def test_shift_0(self):
        data = np.tile(self.reftr, (2, 1))
        dv = pcp.corr_mat_shift(
            data, self.stats, self.reftr, shift_steps=21)
        self.assertListEqual([0, 0], list(dv['value']))

    def test_shift(self):
        s = np.roll(self.reftr, 1)
        data = np.vstack((self.reftr, s))
        dv = pcp.corr_mat_shift(
            data, self.stats, self.reftr, shift_steps=21)
        np.testing.assert_array_equal([0, 1], dv['value'])

    def test_shift_single(self):
        s = np.roll(self.reftr, 1)
        data = np.vstack((self.reftr, s))
        dv = pcp.corr_mat_shift(
            data, self.stats, self.reftr, shift_steps=21)
        np.testing.assert_array_equal([0, 1], dv['value'])


class TestMeasureShift(unittest.TestCase):
    def setUp(self):
        self.stats = CorrStats()
        self.stats.start_lag = -50
        self.stats.npts = 101
        self.stats.delta = 1
        self.stats.corr_start = [UTCDateTime(0), UTCDateTime(100)]
        self.data = np.vstack((np.arange(101), np.arange(101) + 100))

    def test_invalid_sides(self):
        with self.assertRaises(ValueError):
            pcp.measure_shift(None, None, sides='blabla')

    def test_invalid_tw(self):
        with self.assertRaises(ValueError):
            pcp.measure_shift(None, None, tw=[[1, 2, 3]])

    def test_invalid_tw2(self):
        with self.assertRaises(ValueError):
            pcp.measure_shift(
                None, None, tw=[[-1, 2], [2, 3]], sides='both')

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            pcp.measure_shift(
                self.data, None, ref_trc=np.zeros(6))

    @mock.patch('seismic.monitor.post_corr_process.corr_mat_trim')
    @mock.patch('seismic.monitor.post_corr_process.corr_mat_extract_trace')
    @mock.patch('seismic.monitor.post_corr_process.create_shifted_ref_mat')
    @mock.patch(
        'seismic.monitor.post_corr_process.compare_with_modified_reference')
    def test_result0(
        self, cwmr_mock: mock.MagicMock, csrm_mock: mock.MagicMock,
            cmet_mock: mock.MagicMock, cmt_mock: mock.MagicMock):
        cmt_mock.return_value = (self.data, self.stats)
        cmet_mock.return_value = np.ones_like(self.data[1, :]) * 10
        csrm_mock.return_value = np.ones_like(self.data) * 20
        cwmr_mock.return_value = np.ones_like(self.data) * -50
        dtl = pcp.measure_shift(self.data, self.stats)
        cmt_mock.assert_called_once_with(mock.ANY, self.stats, -50, 50)
        np.testing.assert_array_equal(
            cmt_mock.call_args[0][0], self.data)
        cmet_mock.assert_called_once_with(mock.ANY, self.stats)
        np.testing.assert_array_equal(
            cmet_mock.call_args[0][0], self.data)
        csrm_mock.assert_called_once_with(
            mock.ANY, self.stats, mock.ANY)
        np.testing.assert_array_equal(
            csrm_mock.call_args[0][0], np.ones_like(self.data[1, :]) * 10)
        np.testing.assert_array_equal(
            csrm_mock.call_args[0][2], np.linspace(-10, 10, 101))
        np.testing.assert_array_equal(
            cwmr_mock.call_args[0][0], self.data)
        np.testing.assert_array_equal(
            cwmr_mock.call_args[0][1], np.ones_like(self.data) * 20)
        dt = dtl[0]
        self.assertEqual(len(dtl), 1)
        self.assertIsInstance(dt, DV)
        self.assertIsNone(dt.sim_mat)
        np.testing.assert_array_equal(-50, dt.corr)
        np.testing.assert_array_equal(-10, dt.value)
        np.testing.assert_array_equal(['shift'], dt.value_type[0])
        np.testing.assert_array_equal(['absolute_shift'], dt.method[0])
        np.testing.assert_array_equal(
            np.linspace(-10, 10, 101), dt.second_axis)

    @mock.patch('seismic.monitor.post_corr_process.corr_mat_trim')
    @mock.patch('seismic.monitor.post_corr_process.create_shifted_ref_mat')
    @mock.patch(
        'seismic.monitor.post_corr_process.compare_with_modified_reference')
    def test_result1(
        self, cwmr_mock: mock.MagicMock, csrm_mock: mock.MagicMock,
            cmt_mock: mock.MagicMock):
        ref_trc = self.data[0, :] + 5

        cmt_mock.return_value = (np.vstack((self.data, ref_trc)), self.stats)
        csrm_mock.return_value = np.ones_like(self.data) * 20
        cwmr_mock.return_value = np.ones_like(self.data) * -50
        dtl = pcp.measure_shift(
            self.data, self.stats, ref_trc=ref_trc,
            return_sim_mat=True, sides='single')
        cmt_mock.assert_called_once_with(mock.ANY, mock.ANY, -50, 50)
        np.testing.assert_array_equal(
            cmt_mock.call_args[0][0], np.vstack((self.data, ref_trc)))
        csrm_mock.assert_called_once_with(
            mock.ANY, self.stats, mock.ANY)
        np.testing.assert_array_equal(
            csrm_mock.call_args[0][0], ref_trc)
        np.testing.assert_array_equal(
            csrm_mock.call_args[0][2], np.linspace(-10, 10, 101))
        np.testing.assert_array_equal(
            cwmr_mock.call_args[0][0], self.data)
        np.testing.assert_array_equal(
            cwmr_mock.call_args[0][1], np.ones_like(self.data) * 20)
        dt = dtl[0]
        self.assertEqual(len(dtl), 1)
        self.assertIsInstance(dt, DV)
        np.testing.assert_array_equal(
            np.ones_like(self.data) * -50, dt.sim_mat)
        np.testing.assert_array_equal(-50, dt.corr)
        np.testing.assert_array_equal(-10, dt.value)
        np.testing.assert_array_equal(['shift'], dt.value_type[0])
        np.testing.assert_array_equal(['absolute_shift'], dt.method[0])
        np.testing.assert_array_equal(
            np.linspace(-10, 10, 101), dt.second_axis)


class TestApplyShift(unittest.TestCase):
    def setUp(self):
        self.stats = CorrStats()
        self.stats.start_lag = 0
        self.stats.npts = 101
        self.stats.delta = 1
        self.data = np.vstack((np.arange(101), np.arange(101) + 100))

    def test_constant_shift(self):
        shifts = [1, -1]
        outdata = pcp.apply_shift(self.data, self.stats, shifts)
        exp = np.vstack((np.arange(101)+1, np.arange(101) + 99))
        np.testing.assert_array_almost_equal(
            outdata, exp
        )

    def test_linear_shift(self):
        shifts = [np.arange(101), -np.arange(101)]
        outdata = pcp.apply_shift(self.data, self.stats, shifts)
        exp = np.vstack((np.arange(101)*2, np.ones((101,))*100))
        np.testing.assert_array_almost_equal(
            outdata, exp
        )


class TestApplyStretch(unittest.TestCase):
    def setUp(self):
        self.stats = CorrStats()
        self.stats.start_lag = 0
        self.stats.npts = 101
        self.stats.delta = 1
        self.data = np.vstack((
                np.arange(101, dtype=np.float32),
                np.arange(101, dtype=np.float32) + 100))

    def test_constant_stetch(self):
        stretches = [-.05, .05]
        outdata = pcp.apply_stretch(deepcopy(self.data), self.stats, stretches)
        exp = np.vstack((
            np.linspace(0, 105, 101), np.linspace(100, 195, 101)))
        # The taylor series approximation does not deliver the most
        # accurate results. It might make sense to abandon the approximation?
        # For larger stretches this becomes even more inaccurate
        np.testing.assert_array_almost_equal(exp, outdata[0], decimal=1)


if __name__ == "__main__":
    unittest.main()
