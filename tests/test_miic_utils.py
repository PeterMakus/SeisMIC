'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 30th March 2021 01:22:02 pm
Last Modified: Monday, 17th June 2024 04:55:40 pm
'''
from copy import deepcopy
import unittest
import math as mathematics
from unittest import mock
import warnings

import numpy as np
from obspy.core.trace import Stats
from obspy.geodetics import gps2dist_azimuth
from obspy import Inventory, read, UTCDateTime, Trace
from obspy.core import AttribDict
from obspy.core.inventory.network import Network
from obspy.core.inventory.station import Station
from scipy.ndimage import convolve1d

from seismic.utils.fetch_func_from_str import func_from_str
import seismic.utils.miic_utils as mu
from seismic.correlate.stats import CorrStats


class TestBazCalc(unittest.TestCase):
    def setUp(self) -> None:
        self.latitude1 = 10
        self.longitude1 = 15
        self.latitude2 = 12
        self.longitude2 = -2
        stat = Station('BLA', self.latitude1, self.longitude1, 0)
        net = Network('T1', stations=[stat])
        self.inv1 = Inventory(networks=[net])
        stat = Station('BLU', self.latitude2, self.longitude2, 0)
        net = Network('T1', stations=[stat])
        self.inv2 = Inventory(networks=[net])
        self.dist, self.az, self.baz = gps2dist_azimuth(
            self.latitude1, self.longitude1, self.latitude2, self.longitude2)
        self.st1 = AttribDict(
            {'stla': self.latitude1, 'stlo': self.longitude1})
        self.st2 = AttribDict(
            {'stla': self.latitude2, 'stlo': self.longitude2})

    def test_result_inv(self):
        az, baz, dist = mu.inv_calc_az_baz_dist(self.inv1, self.inv2)
        self.assertEqual(az, self.az)
        self.assertEqual(baz, self.baz)
        self.assertEqual(dist, self.dist)

    def test_identical_coords(self):
        _, _, dist = mu.inv_calc_az_baz_dist(self.inv1, self.inv1)
        self.assertEqual(dist, 0)

    def test_result_tr(self):
        az, baz, dist = mu.trace_calc_az_baz_dist(self.st1, self.st2)
        self.assertEqual(az, self.az)
        self.assertEqual(baz, self.baz)
        self.assertEqual(dist, self.dist)


class TestFilterStat_Dist(unittest.TestCase):
    @mock.patch('seismic.utils.miic_utils.inv_calc_az_baz_dist')
    def test_result_True(self, calc_dist_mock: mock.MagicMock):
        dist = np.random.randint(100, 1e6)  # in m
        calc_dist_mock.return_value = (
            np.random.randint(0, 360), np.random.randint(0, 360), dist)
        queried_dist = dist + np.random.randint(10, 1000)
        self.assertTrue(mu.filter_stat_dist('bla', 'blub', queried_dist))

    @mock.patch('seismic.utils.miic_utils.inv_calc_az_baz_dist')
    def test_result_False(self, calc_dist_mock: mock.MagicMock):
        dist = np.random.randint(100, 1e6)  # in m
        calc_dist_mock.return_value = (
            np.random.randint(0, 360), np.random.randint(0, 360), dist)
        queried_dist = dist - np.random.randint(10, 1000)
        self.assertFalse(mu.filter_stat_dist('bla', 'blub', queried_dist))


class TestResampleOrDecimate(unittest.TestCase):
    def test_decimate(self):
        st = read()
        freq_new = st[0].stats.sampling_rate//4
        st_filt = mu.resample_or_decimate(st, freq_new)
        self.assertEqual(st_filt[0].stats.sampling_rate, freq_new)
        self.assertIn('decimate', st_filt[0].stats.processing[-1])
        self.assertIn('filter', st_filt[0].stats.processing[-2])

    def test_resample(self):
        st = read()
        freq_new = st[0].stats.sampling_rate/2.5
        st_filt = mu.resample_or_decimate(st, freq_new, filter=False)
        self.assertEqual(st_filt[0].stats.sampling_rate, freq_new)
        self.assertIn('resample', st_filt[0].stats.processing[-1])
        self.assertIn('no_filter=True', st_filt[0].stats.processing[-1])

    def test_filter_w_high_factor(self):
        # Here we filter 10% lower than Nyquist
        st = read()
        freq_new = 1
        st_filt = mu.resample_or_decimate(st, freq_new)
        self.assertEqual(st_filt[0].stats.sampling_rate, freq_new)
        self.assertIn('decimate', st_filt[0].stats.processing[-1])
        self.assertIn('filter', st_filt[0].stats.processing[-2])
        self.assertIn(
            "filter(options={'freq': 0.45, 'maxorder': 12}:"
            + ":type='lowpass_cheby_2')", st_filt[0].stats.processing[-2])

    def test_new_freq_higher_than_native(self):
        st = read()
        freq_new = st[0].stats.sampling_rate+5
        with self.assertRaises(ValueError):
            _ = mu.resample_or_decimate(st, freq_new, filter=False)

    def test_st_differing_sr(self):
        st = read()
        st[0].decimate(2)
        st[1].decimate(4)
        freq_new = st[0].stats.sampling_rate
        with self.assertWarns(UserWarning):
            st_out = mu.resample_or_decimate(st, freq_new, filter=False)
        self.assertEqual(st_out[2].stats.sampling_rate, freq_new)
        self.assertEqual(st_out[0].stats.sampling_rate, freq_new)
        self.assertEqual(st_out[1].stats.sampling_rate, freq_new/2)


class TestTrimTraceDelta(unittest.TestCase):
    def setUp(self):
        self.tr = read()[0]

    def test_times(self):
        delta = np.random.randint(1, 10)
        tr = mu.trim_trace_delta(self.tr.copy(), delta, delta)
        self.assertEqual(tr.stats.starttime, self.tr.stats.starttime+delta)
        self.assertEqual(tr.stats.endtime, self.tr.stats.endtime-delta)

    def test_delta_0(self):
        tr = mu.trim_trace_delta(self.tr.copy(), 0, 0)
        self.assertEqual(tr.stats.starttime, self.tr.stats.starttime)
        self.assertEqual(tr.stats.endtime, self.tr.stats.endtime)

    def test_pass_kwargs(self):
        delta = np.random.randint(0, 5)
        # this is alright as long as no error is raised
        _ = mu.trim_trace_delta(self.tr.copy(), delta, delta, fill_value=0)


class TestHeaderToNPArray(unittest.TestCase):
    def test_result(self):
        st = Stats({
            'starttime': UTCDateTime(0),
            'endtime': UTCDateTime(10),
            'corr_start': UTCDateTime(0),
            'corr_end': UTCDateTime(10),
            'other': 'blub'})
        exp = dict(st)
        for key in exp:
            if isinstance(exp[key], UTCDateTime):
                exp[key] = exp[key].timestamp
            exp[key] = np.array([exp[key]])

        d = mu.save_header_to_np_array(st)
        self.assertDictEqual(d, exp)

    def test_list(self):
        st = CorrStats({
            'starttime': [UTCDateTime(ii) for ii in range(10)],
            'endtime': [UTCDateTime(ii+10) for ii in range(10)],
            'corr_start': [UTCDateTime(ii) for ii in range(10)],
            'corr_end': [UTCDateTime(ii+10) for ii in range(10)],
            'other': 'blub'})
        exp = dict(st)
        for key in exp:
            if isinstance(exp[key], list):
                exp[key] = np.array([x.timestamp for x in exp[key]])
            else:
                exp[key] = np.array([exp[key]])

        d = mu.save_header_to_np_array(st)
        for key in d:
            self.assertTrue(np.all(d[key] == exp[key]))


class TestLoadHeaderFromNPArray(unittest.TestCase):
    def setUp(self):
        self.st = Stats({
            'starttime': UTCDateTime(0),
            'endtime': UTCDateTime(10),
            'corr_start': UTCDateTime(0),
            'corr_end': UTCDateTime(10),
            'other': 'blub'})

    def test_integral(self):
        d = mu.save_header_to_np_array(self.st)
        st = mu.load_header_from_np_array(d)
        self.assertDictEqual(st, dict(self.st))

    def test_list(self):
        st = CorrStats({
            'starttime': [UTCDateTime(ii) for ii in range(10)],
            'endtime': [UTCDateTime(ii+10) for ii in range(10)],
            'corr_start': [UTCDateTime(ii) for ii in range(10)],
            'corr_end': [UTCDateTime(ii+10) for ii in range(10)],
            'other': 'blub'})
        d = mu.save_header_to_np_array(st)
        st2 = mu.load_header_from_np_array(d)
        self.assertDictEqual(st2, dict(st))


class ConvertUTCToTimeStamp(unittest.TestCase):
    def test_single(self):
        inp = UTCDateTime(0)
        exp = np.zeros(1)
        self.assertTrue(np.all(
            exp == mu.convert_utc_to_timestamp(inp)))

    def test_list(self):
        inp = [UTCDateTime(ii) for ii in range(10)]
        exp = np.arange(0, 10)
        self.assertTrue(np.all(
            exp == mu.convert_utc_to_timestamp(inp)))


class ConvertTimeStampToUTCDT(unittest.TestCase):
    def test_single(self):
        exp = UTCDateTime(0)
        inp = np.zeros(1)
        self.assertTrue(np.all(
            exp == mu.convert_timestamp_to_utcdt(inp)))

    def test_list(self):
        exp = [UTCDateTime(ii) for ii in range(10)]
        inp = np.arange(0, 10)
        self.assertTrue(np.all(
            exp == mu.convert_timestamp_to_utcdt(inp)))


class TestGetValidTraces(unittest.TestCase):
    def test_norm_stream(self):
        st = read()
        out = st.copy()
        mu.get_valid_traces(out)
        self.assertEqual(st, out)

    def test_mask_tr(self):
        stin = read()
        st = stin.copy()
        da = np.ma.masked_values([1, 1, 1, 1], 1)
        st.append(Trace(da))
        mu.get_valid_traces(st)
        self.assertEqual(stin.count(), st.count())
        self.assertEqual(stin, st)


class TestDiscardShortTraces(unittest.TestCase):
    def test_result(self):
        st = read()
        stin = st.copy()
        stin.append(Trace(np.empty(25)))
        stin[3].stats.delta = st[0].stats.delta
        mu.discard_short_traces(stin, 10)
        self.assertEqual(st.count(), stin.count())
        self.assertEqual(st, stin)


class TestNanMovingAv(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.random.random((500,))
        self.exp = convolve1d(self.data, np.ones(101))/101

    def test_result(self):
        out = mu.nan_moving_av(self.data, 50)
        # There will be a slight difference at the ends of the array
        # as scipy will be less precise here (the weighting is not correct)
        # The first 50 elements
        np.testing.assert_allclose(self.exp[:50], out[:50], rtol=1e-1)
        np.testing.assert_allclose(self.exp[-50:], out[-50:], rtol=1e-1)
        np.testing.assert_allclose(self.exp[50:-50], out[50:-50])

    def test_result_nan(self):
        # test the same with nans
        data = deepcopy(self.data)
        data[125] = np.nan
        out = mu.nan_moving_av(data, 50)
        # Changing one element should be covered by 2% tolerance
        np.testing.assert_allclose(self.exp[50:-50], out[50:-50], rtol=2e-2)

    def test_axis2(self):
        data = np.random.random((25, 25, 25))
        for ax in [0, 1, 2]:
            exp = convolve1d(data, np.ones(11), axis=ax)/11
            out = mu.nan_moving_av(data, 5, axis=ax)
            np.testing.assert_allclose(
                exp.swapaxes(0, ax)[5:-5], out.swapaxes(0, ax)[5:-5])
            # Let's ignore the edge effects for now


class TestStreamRequireDtype(unittest.TestCase):
    def test_result(self):
        st = read()
        self.assertNotEqual(st[0].data.dtype, np.float32)
        mu.stream_require_dtype(st, np.float32)
        self.assertEqual(st[0].data.dtype, np.float32)


class TestFunctionFromString(unittest.TestCase):
    def test_not_existent(self):
        with self.assertRaises(ModuleNotFoundError):
            func_from_str('this.module.does.not.exist')

    def test_not_an_attrib(self):
        with self.assertRaises(AttributeError):
            func_from_str('math.nonsensefunct')

    def test_result(self):
        funct = func_from_str('math.sqrt')
        self.assertEqual(funct, mathematics.sqrt)


class TestCorrectPolarity(unittest.TestCase):
    def setUp(self):
        self.st = read()
        self.inv = mock.MagicMock()

    def test_flip(self):
        self.inv.get_orientation.return_value = {
            'dip': 90,
            'aximuth': 0
        }
        st_correct = self.st.copy()
        mu.correct_polarity(st_correct, self.inv)
        self.inv.get_orientation.assert_called_with(
            st_correct.select(component="Z")[0].id,
            datetime=st_correct.select(component="Z")[0].stats.starttime)
        np.testing.assert_array_equal(
            st_correct.select(component="Z")[0].data,
            -1 * self.st.select(component="Z")[0].data
        )

    def test_no_flip(self):
        self.inv.get_orientation.return_value = {
            'dip': -90,
            'aximuth': 0
        }
        st_correct = self.st.copy()
        mu.correct_polarity(st_correct, self.inv)
        self.inv.get_orientation.assert_called_with(
            st_correct.select(component="Z")[0].id,
            datetime=st_correct.select(component="Z")[0].stats.starttime)
        np.testing.assert_array_equal(
            st_correct.select(component="Z")[0].data,
            self.st.select(component="Z")[0].data
        )


class TestNanHelper(unittest.TestCase):
    def test_result(self):
        # create 1d array that holds nans
        y = np.arange(15, dtype=float)
        y[11:13] = y[3] = np.nan
        nans, x = mu.nan_helper(y)
        np.testing.assert_array_equal(True, nans[nans])
        np.testing.assert_array_equal(False, ~nans[nans])
        np.testing.assert_array_equal(
            x(nans), [3, 11, 12]
        )
        np.testing.assert_array_equal(
            x(~nans), np.hstack((np.arange(3), np.arange(4, 11), [13, 14]))
        )

    def test_no_nans(self):
        y = np.arange(15)
        nans, x = mu.nan_helper(y)
        np.testing.assert_array_equal(False, nans)
        np.testing.assert_array_equal(
            x(nans), []
        )


class TestGapHandler(unittest.TestCase):
    @mock.patch('seismic.utils.miic_utils.interpolate_gaps_st')
    @mock.patch('seismic.utils.miic_utils.cos_taper_st')
    def test_nothing_to_do(
            self, ct_mock: mock.MagicMock, igst_mock: mock.MagicMock):
        st = read()
        ct_mock.return_value = st
        igst_mock.return_value = st
        with mock.patch.multiple(
            st, merge=mock.MagicMock(return_value=st),
            split=mock.MagicMock(return_value=st),
        ):
            out = mu.gap_handler(st, 20, 100, 15)
            ct_mock.assert_called_once_with(st, 15, False, False)
            igst_mock.assert_called_once_with(st, max_gap_len=20)
            st.merge.assert_has_calls(
                [mock.call(method=-1), mock.call()]
            )
            self.assertEqual(out, st)


class TestInterpolateGaps(unittest.TestCase):
    def test_not_masked_no_nan(self):
        A = np.ones(5)
        np.testing.assert_array_equal(A, mu.interpolate_gaps(A, 10))

    def test_length_above_threshold(self):
        x = np.arange(50, dtype=float)
        x[5:35] = np.nan
        x[38:43] = np.nan
        with warnings.catch_warnings(record=True) as w:
            out = mu.interpolate_gaps(deepcopy(x), max_gap_len=4)
            np.testing.assert_array_equal(
                x, out)
            self.assertNotEqual(len(w), 0)
        self.assertTrue(np.ma.is_masked(out))

    def test_interpolate_one_skip_one(self):
        x = np.arange(50, dtype=float)
        x[5:35] = np.nan
        x[38:43] = np.nan
        with warnings.catch_warnings(record=True) as w:
            out = mu.interpolate_gaps(deepcopy(x), max_gap_len=7)
            self.assertNotEqual(len(w), 0)
        np.testing.assert_array_equal(
            True, np.isnan(out[5:35]))
        np.testing.assert_array_equal(
            False, np.isnan(out[35:]))
        self.assertTrue(np.ma.is_masked(out))

    def test_interpolate_all(self):
        x = np.arange(50, dtype=float)
        x[5:35] = np.nan
        x[38:43] = np.nan
        with warnings.catch_warnings(record=True) as w:
            out = mu.interpolate_gaps(deepcopy(x), max_gap_len=-1)
            self.assertEqual(len(w), 0)
        np.testing.assert_array_equal(
            False, np.isnan(out))
        self.assertFalse(np.ma.is_masked(out))


class TestInterpolateGapsSt(unittest.TestCase):
    @mock.patch('seismic.utils.miic_utils.interpolate_gaps')
    def test_result(self, ig_mock: mock.MagicMock):
        st = read()
        ig_mock.return_value = np.zeros(5)
        out = mu.interpolate_gaps_st(st.copy(), 15)
        calls = [mock.call(mock.ANY, 15) for tr in st]
        ig_mock.assert_has_calls(calls)
        for tr in out:
            np.testing.assert_array_equal(tr.data, np.zeros(5))
        for cal, tr in zip(ig_mock.call_args_list, st):
            np.testing.assert_array_equal(tr.data, cal[0][0])


class TestSortCombinationsAlphabetically(unittest.TestCase):
    def test_sort_combinations(self):
        netcomb = "NET1-NET2"
        stacomb = "STA2-STA1"
        loccomb = "LOC1-LOC2"
        chacomb = "CHA2-CHA1"
        expected_netcomb = "NET1-NET2"
        expected_stacomb = "STA2-STA1"
        expected_loccomb = "LOC1-LOC2"
        expected_chacomb = "CHA2-CHA1"

        sorted_netcomb, sorted_stacomb, sorted_loccomb, sorted_chacomb = mu.sort_combinations_alphabetically(
            netcomb, stacomb, loccomb, chacomb)

        self.assertEqual(sorted_netcomb, expected_netcomb)
        self.assertEqual(sorted_stacomb, expected_stacomb)
        self.assertEqual(sorted_loccomb, expected_loccomb)
        self.assertEqual(sorted_chacomb, expected_chacomb)

    def test_sort_combinations2(self):
        netcomb = "NET2-NET1"
        stacomb = "STA2-STA1"
        loccomb = "LOC1-LOC2"
        chacomb = "CHA2-CHA1"
        expected_netcomb = "NET1-NET2"
        expected_stacomb = "STA1-STA2"
        expected_loccomb = "LOC2-LOC1"
        expected_chacomb = "CHA1-CHA2"

        sorted_netcomb, sorted_stacomb, sorted_loccomb, sorted_chacomb = mu.sort_combinations_alphabetically(
            netcomb, stacomb, loccomb, chacomb)

        self.assertEqual(sorted_netcomb, expected_netcomb)
        self.assertEqual(sorted_stacomb, expected_stacomb)
        self.assertEqual(sorted_loccomb, expected_loccomb)
        self.assertEqual(sorted_chacomb, expected_chacomb)


if __name__ == "__main__":
    unittest.main()
