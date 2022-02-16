'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 30th March 2021 01:22:02 pm
Last Modified: Wednesday, 16th February 2022 10:07:03 am
'''
from copy import deepcopy
import unittest
import math as mathematics
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
