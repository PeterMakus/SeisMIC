'''
UnitTests for the preprocess module.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 15th March 2021 11:14:04 am
Last Modified: Monday, 15th March 2021 03:21:14 pm
'''
import unittest

import numpy as np
from obspy import read
from obspy.core.inventory.inventory import read_inventory

from miic3.trace_data.preprocess import Preprocessor, FrequencyError
from miic3.trace_data.waveform import Store_Client


class TestPreprocessor(unittest.TestCase):
    """
    Test :class:`~miic3.trace_data.Preprocessor`.
    """
    def setUp(self) -> None:
        self.store_client = Store_Client('IRIS', '/', read_only=True)

    def test_aliasing_protect(self):
        """
        When the sampling frequency is too high, this should raise an Error.
        """
        with self.assertRaises(AssertionError):
            Preprocessor(
                self.store_client, filter=(.1, 100), sampling_rate=100,
                outfolder='x')

    def test_frequency_error(self):
        """
        Test the lowest level preprocessing frequency error
        """
        p = Preprocessor(
                self.store_client, filter=(.1, 10), sampling_rate=1000,
                outfolder='x')
        with self.assertRaises(FrequencyError):
            p._preprocess(read())

    def test_preprocess(self):
        """
        Test whether the actual low-level preprocessing function works
        """
        p = Preprocessor(
                self.store_client, filter=(.1, 10), sampling_rate=25,
                outfolder='x')
        st, _ = p._preprocess(read(), inv=read_inventory())
        self.assertIsInstance(st[0].data[0], np.float32)
        self.assertEqual(st[0].stats.sampling_rate, 25)


if __name__ == "__main__":
    unittest.main()
