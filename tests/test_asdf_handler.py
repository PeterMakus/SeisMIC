'''
Module to test the asdf handler.
Author: Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 18th March 2021 04:26:31 pm
Last Modified: Thursday, 18th March 2021 04:34:50 pm
'''
import unittest

import random
from obspy import read, read_inventory
from pyasdf import ASDFDataSet


class TestNoiseDB(unittest.TestCase):
    def setUp(self) -> None:
        # Example data
        st = read()
        inv = read_inventory
        a = random.random()
        self.filter = (a, 3*a)
        self.sampling_rate = 7*a
        self.station = st[0].stats.station
        self.network = st[0].stats.network

        with ASDFDataSet('test.h5') as ds:
            ds.add_waveforms(st, tag='processed')
            ds.add_stationxml(inv)
