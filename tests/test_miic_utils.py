'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 30th March 2021 01:22:02 pm
Last Modified: Tuesday, 20th July 2021 04:09:49 pm
'''
import unittest
import math as mathematics

import numpy as np
from obspy.geodetics import gps2dist_azimuth
from obspy import Inventory, read
from obspy.core import AttribDict
from obspy.core.inventory.network import Network
from obspy.core.inventory.station import Station

from miic3.utils.fetch_func_from_str import func_from_str
from miic3.utils.miic_utils import trace_calc_az_baz_dist,\
    inv_calc_az_baz_dist, resample_or_decimate
import miic3.utils.miic_utils as mu


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
        az, baz, dist = inv_calc_az_baz_dist(self.inv1, self.inv2)
        self.assertEqual(az, self.az)
        self.assertEqual(baz, self.baz)
        self.assertEqual(dist, self.dist)

    def test_identical_coords(self):
        _, _, dist = inv_calc_az_baz_dist(self.inv1, self.inv1)
        self.assertEqual(dist, 0)

    def test_result_tr(self):
        az, baz, dist = trace_calc_az_baz_dist(self.st1, self.st2)
        self.assertEqual(az, self.az)
        self.assertEqual(baz, self.baz)
        self.assertEqual(dist, self.dist)


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


class TestResampleOrDecimate(unittest.TestCase):
    def test_decimate(self):
        st = read()
        freq_new = st[0].stats.sampling_rate//4
        st_filt = resample_or_decimate(st, freq_new)
        self.assertEqual(st_filt[0].stats.sampling_rate, freq_new)
        self.assertIn('decimate', st_filt[0].stats.processing[-1])
        self.assertIn('filter', st_filt[0].stats.processing[-2])

    def test_resample(self):
        st = read()
        freq_new = st[0].stats.sampling_rate/2.5
        st_filt = resample_or_decimate(st, freq_new, filter=False)
        self.assertEqual(st_filt[0].stats.sampling_rate, freq_new)
        self.assertIn('resample', st_filt[0].stats.processing[-1])
        self.assertIn('no_filter=True', st_filt[0].stats.processing[-1])

    def test_new_freq_higher_than_native(self):
        st = read()
        freq_new = st[0].stats.sampling_rate+5
        with self.assertRaises(ValueError):
            _ = resample_or_decimate(st, freq_new, filter=False)


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


if __name__ == "__main__":
    unittest.main()
