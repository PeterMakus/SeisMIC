'''
:copyright:
:license:
   `GNU Lesser General Public License, Version 3 <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 04:07:16 pm
Last Modified: Tuesday, 27th July 2021 10:56:17 am
'''

import unittest

import numpy as np
from obspy.core import AttribDict
from obspy import read, Stream, Trace

from seismic.correlate import preprocessing_stream as ppst


class TestCosTaper(unittest.TestCase):
    #
    def setUp(self):
        self.sr = 10  # sampling rate
        st = AttribDict({'sampling_rate': self.sr})
        self.testtr = Trace(np.ones(1000), header=st)
        tl = np.random.randint(1, high=20)
        self.tls = tl * self.sr  # taper len in samples
        self.tr_res = ppst.cos_taper(self.testtr.copy(), tl, False)

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
            ppst.cos_taper(self.testtr.copy(), np.random.randint(-100, 0), False)
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


if __name__ == "__main__":
    unittest.main()
