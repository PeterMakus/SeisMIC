'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 04:07:16 pm
Last Modified: Wednesday, 9th February 2022 01:58:21 pm
'''

import unittest
from unittest import mock
import warnings

import numpy as np
from obspy.core import AttribDict
from obspy import read, Stream, Trace

from seismic.correlate import preprocessing_stream as ppst


class TestCosTaperSt(unittest.TestCase):
    def setUp(self) -> None:
        self.st = read()

    @mock.patch('seismic.correlate.preprocessing_stream.cos_taper')
    def test_result(self, cos_taper_mock):
        trcs = [
            Trace(np.zeros((10,))), Trace(np.ones((10,))),
            Trace(2*np.ones((10,)))]
        cos_taper_mock.side_effect = trcs
        exp = Stream(trcs)
        out = ppst.cos_taper_st(self.st.copy(), 5, True)
        calls = [
            mock.call(self.st[0], 5, True), mock.call(self.st[1], 5, True),
            mock.call(self.st[2], 5, True)]
        cos_taper_mock.assert_has_calls(calls)
        for tr, tro in zip(exp, out):
            np.testing.assert_array_equal(tr.data, tro.data)

    @mock.patch('seismic.correlate.preprocessing_stream.cos_taper')
    def test_trace_cannot_be_tapered(self, cos_taper_mock):
        intr = Trace(np.zeros((10,)))
        cos_taper_mock.side_effect = ValueError
        with warnings.catch_warnings(record=True) as w:
            out = ppst.cos_taper_st(intr, 5, True)
            self.assertEqual(len(w), 1)
        calls = [
            mock.call(intr, 5, True)]
        cos_taper_mock.assert_has_calls(calls)
        self.assertEqual(out, Stream([intr]))


class TestCosTaper(unittest.TestCase):
    def setUp(self):
        self.sr = 10  # sampling rate
        st = AttribDict({'sampling_rate': self.sr})
        self.testtr = Trace(np.ones(1000), header=st)
        tl = np.random.randint(1, high=20)
        self.tls = tl * self.sr  # taper len in samples
        self.tr_res = ppst.cos_taper(self.testtr.copy(), tl, False)

    def test_in_place(self):
        self.sr = 10  # sampling rate
        st = AttribDict({'sampling_rate': self.sr})
        testtro = Trace(np.ones(1000), header=st)
        testtr = testtro.copy()
        self.assertEqual(testtr, testtro)
        ppst.cos_taper(testtr, 5, False)
        self.assertNotEqual(testtr, testtro)

    def test_ends(self):
        # Check that ends reduce to 0
        self.assertAlmostEqual(self.tr_res.data[0], 0)
        self.assertAlmostEqual(self.tr_res.data[-1], 0)

    def test_middle(self):
        # Assert that the rest (in the middle) stayed the same
        self.assertTrue(np.array_equal(
            self.testtr[self.tls:-self.tls], self.tr_res[self.tls:-self.tls]))

    def test_up_down(self):
        # Everything else should be between 1 and 0
        # up
        self.assertTrue(np.all(self.tr_res[1:-1] > 0))
        self.assertTrue(np.all(self.tr_res[1:self.tls] < 1))
        # down
        self.assertTrue(np.all(self.tr_res[-self.tls:-1] < 1))

    def test_empty_trace(self):
        testtr = Trace(np.array([]), header=self.testtr.stats)
        with self.assertRaises(ValueError):
            ppst.cos_taper(testtr, 10, False)

    def test_invalid_taper_len(self):
        with self.assertRaises(ValueError):
            ppst.cos_taper(
                self.testtr.copy(), np.random.randint(-100, 0), False)
        with self.assertRaises(ValueError):
            ppst.cos_taper(self.testtr.copy(), 501*self.sr, False)

    def test_masked_value(self):
        tr0 = read()[0]
        tr1 = tr0.copy()
        tr1.stats.starttime += 240
        st = Stream([tr0, tr1])
        tr = st.merge()[0]
        tl = np.random.randint(1, high=5)
        ttr = ppst.cos_taper(tr, tl, True)
        # Check that ends reduce to 0
        self.assertAlmostEqual(ttr.data[0], 0)
        self.assertAlmostEqual(ttr.data[-1], 0)
        self.assertAlmostEqual(ttr.data[tr0.count()-1], 0)
        self.assertAlmostEqual(ttr.data[-tr1.count()], 0)
        # Also the mask should be retained
        self.assertEqual(
            len(ttr.data[ttr.data.mask]), ttr.count()-tr0.count()-tr1.count())


class TestDetrendSt(unittest.TestCase):
    # This happens inplace
    def test_result_no_gaps(self):
        data = Stream([Trace(np.ones((25,)))])
        ppst.detrend_st(data, **{'type': 'constant'})
        np.testing.assert_allclose(data[0].data, 0)

    def test_result_gaps(self):
        tr0 = read()[0]
        tr0.data = np.ones_like(tr0.data)
        tr1 = tr0.copy()
        tr1.stats.starttime = tr0.stats.endtime + 5
        data = Stream([tr0, tr1]).merge()
        ppst.detrend_st(data, **{'type': 'constant'})
        self.assertEqual(data.count(), 1)
        np.testing.assert_allclose(data[0].data, 0)


class TestStFilter(unittest.TestCase):
    def setUp(self):
        self.st = read()

    def test_not_a_stream(self):
        with self.assertRaises(TypeError):
            ppst.stream_filter('bla', 'bla', {})

    def test_result_no_gap(self):
        data = self.st[0].copy()
        out = ppst.stream_filter(
            data, 'bandpass', {'freqmin': 0.0005, 'freqmax': 0.001})
        np.testing.assert_allclose(out[0], 0, atol=1e-3)

    def test_result_gaps(self):
        tr0 = self.st[0].copy()
        tr1 = tr0.copy()
        tr1.stats.starttime = tr0.stats.endtime + 5
        data = Stream([tr0, tr1]).merge()
        data.append(self.st[1])
        out = ppst.stream_filter(
            data, 'bandpass', {'freqmin': 0.0005, 'freqmax': 0.001})
        self.assertEqual(data.count(), 2)
        for tr in out:
            np.testing.assert_allclose(tr.data, 0, atol=1e-3)


class TestStreamMaskAtUTC(unittest.TestCase):
    def setUp(self):
        self.st = read()

    def test_end_and_len_None(self):
        with self.assertRaises(ValueError):
            ppst.stream_mask_at_utc(self.st, [1, 2, 3])

    def test_end_and_len_defined(self):
        with self.assertRaises(ValueError):
            ppst.stream_mask_at_utc(self.st, [1, 2, 3], [3, 4, 5], 2)

    def test_end_and_start_diff_len(self):
        with self.assertRaises(ValueError):
            ppst.stream_mask_at_utc(self.st, [1, 2, 3], [3, 4, 5, 8])

    @mock.patch('seismic.correlate.preprocessing_stream.trace_mask_at_utc')
    def test_masklen(self, tr_mask_mock):
        length = np.random.randint(1, 5)
        starts = np.array([self.st[0].stats.starttime]*2) + np.array([1, 7])
        calls = []
        for tr in self.st:
            calls.append(mock.call(tr, mock.ANY, mock.ANY))
        ppst.stream_mask_at_utc(self.st, starts, masklen=length)
        tr_mask_mock.assert_has_calls(calls)
        np.testing.assert_array_equal(
            tr_mask_mock.call_args_list[-1][-2][1], starts)
        np.testing.assert_array_equal(
            tr_mask_mock.call_args_list[-2][-2][1], starts)
        np.testing.assert_array_equal(
            tr_mask_mock.call_args_list[-1][-2][2], starts+length)
        np.testing.assert_array_equal(
            tr_mask_mock.call_args_list[-2][-2][2], starts+length)

    @mock.patch('seismic.correlate.preprocessing_stream.trace_mask_at_utc')
    def test_maskend(self, tr_mask_mock):
        length = np.array([np.random.randint(1, 5), np.random.randint(1, 5)])
        starts = np.array([self.st[0].stats.starttime]*2) + np.array([1, 7])
        ends = starts + length
        calls = []
        for tr in self.st:
            calls.append(mock.call(tr, mock.ANY, mock.ANY))
        ppst.stream_mask_at_utc(self.st, starts, ends=ends)
        tr_mask_mock.assert_has_calls(calls)
        np.testing.assert_array_equal(
            tr_mask_mock.call_args_list[-1][-2][1], starts)
        np.testing.assert_array_equal(
            tr_mask_mock.call_args_list[-2][-2][1], starts)
        np.testing.assert_array_equal(
            tr_mask_mock.call_args_list[-1][-2][2], ends)
        np.testing.assert_array_equal(
            tr_mask_mock.call_args_list[-2][-2][2], ends)


class TestTraceMaskAtUTC(unittest.TestCase):
    def setUp(self):
        self.tr = read()[0]

    def test_mask_inside(self):
        startsdelta = np.array([1, 10]) + np.array(
                [np.random.randint(0, 5), np.random.randint(0, 5)])
        starts = np.array([self.tr.stats.starttime]*2) \
            + startsdelta
        masklen = np.array([np.random.randint(1, 4), np.random.randint(1, 4)])
        ends = starts + masklen
        ppst.trace_mask_at_utc(self.tr, starts, ends)
        self.assertTrue(np.ma.is_masked(self.tr.data))
        self.assertFalse(np.ma.is_masked(
            self.tr.data[:int(startsdelta[0]*self.tr.stats.sampling_rate)]))
        self.assertFalse(np.ma.is_masked(
            self.tr.data[int((
                startsdelta[-1]+masklen[-1])*self.tr.stats.sampling_rate)+1:]))

    def test_mask_at_start(self):
        startsdelta = -10 + np.random.randint(0, 5)
        starts = np.array([self.tr.stats.starttime+startsdelta])
        masklen = np.random.randint(11, 15)
        ends = starts + masklen
        ppst.trace_mask_at_utc(self.tr, starts, ends)
        self.assertTrue(np.ma.is_masked(self.tr.data))
        self.assertFalse(np.ma.is_masked(
            self.tr.data[int((
                startsdelta+masklen)*self.tr.stats.sampling_rate)+1:]))

    def test_mask_at_end(self):
        startsdelta = -10 + np.random.randint(0, 5)
        starts = np.array([self.tr.stats.endtime+startsdelta])
        masklen = np.random.randint(11, 15)
        ends = starts + masklen
        ppst.trace_mask_at_utc(self.tr, starts, ends)
        self.assertTrue(np.ma.is_masked(self.tr.data))
        self.assertFalse(np.ma.is_masked(
            self.tr.data[:int(startsdelta*self.tr.stats.sampling_rate-2)]))


if __name__ == "__main__":
    unittest.main()
