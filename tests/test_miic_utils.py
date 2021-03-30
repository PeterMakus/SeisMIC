'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 30th March 2021 01:22:02 pm
Last Modified: Tuesday, 30th March 2021 01:49:58 pm
'''
import unittest

from obspy.geodetics import gps2dist_azimuth
from obspy import Inventory, Trace
from obspy.core import AttribDict
from obspy.core.inventory.network import Network
from obspy.core.inventory.station import Station

from miic3.utils.miic_utils import trace_calc_az_baz_dist, inv_calc_az_baz_dist


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
        st = AttribDict(sac={'stla': self.latitude1, 'stlo': self.longitude1})
        self.tr1 = Trace(header=st)
        st = AttribDict(sac={'stla': self.latitude2, 'stlo': self.longitude2})
        self.tr2 = Trace(header=st)

    def test_result_inv(self):
        az, baz, dist = inv_calc_az_baz_dist(self.inv1, self.inv2)
        self.assertEqual(az, self.az)
        self.assertEqual(baz, self.baz)
        self.assertEqual(dist, self.dist)

    def test_result_tr(self):
        az, baz, dist = trace_calc_az_baz_dist(self.tr1, self.tr2)
        self.assertEqual(az, self.az)
        self.assertEqual(baz, self.baz)
        self.assertEqual(dist, self.dist)


if __name__ == "__main__":
    unittest.main()
