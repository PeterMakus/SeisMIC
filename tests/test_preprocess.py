'''
UnitTests for the preprocess module.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 15th March 2021 11:14:04 am
Last Modified: Thursday, 27th May 2021 04:19:48 pm
'''
import unittest
from unittest import mock

import numpy as np
from obspy import read
from obspy.core.inventory.inventory import read_inventory

from miic3.trace_data.preprocess import Preprocessor, FrequencyError
from miic3.trace_data.waveform import Store_Client


# This module is rather difficult to test


class TestPreprocessor(unittest.TestCase):
    """
    Test :class:`~miic3.trace_data.Preprocessor`.
    """
    def setUp(self) -> None:
        self.store_client = Store_Client('IRIS', '/', read_only=True)

    def test_frequency_error(self):
        """
        Test the lowest level preprocessing frequency error
        """
        p = Preprocessor(
                self.store_client, sampling_rate=1000,
                outfolder='x', remove_response=True)
        with self.assertRaises(FrequencyError):
            p._preprocess(read(), inv=read_inventory(), taper_len=0)

    def test_preprocess(self):
        """
        Test whether the actual low-level preprocessing function works
        """
        p = Preprocessor(
                self.store_client, sampling_rate=25,
                outfolder='x', remove_response=True)
        st, _ = p._preprocess(read(), inv=read_inventory(), taper_len=3)
        # right data type?
        self.assertIsInstance(st[0].data[0], np.float32)
        # new sampling frequency
        self.assertEqual(st[0].stats.sampling_rate, 25)
        # Response removal works?
        self.assertTrue('response' in st[0].stats.processing[-1])

    @mock.patch('miic3.trace_data.preprocess.os.path.exists')
    def test_path_exist(self, path_exists_mock):
        path_exists_mock.return_value = True
        p = Preprocessor(
                self.store_client, sampling_rate=25,
                outfolder='x', remove_response=True)
        with self.assertRaises(PermissionError):
            p.preprocess('bla', 'blub', '*', '*')

    def test_use_bulk(self):
        p = Preprocessor(
                self.store_client, sampling_rate=25,
                outfolder='x', remove_response=True)
        with self.assertRaises(TypeError):
            p.preprocess('bla', ['1', '2'], '*', '*')
        with self.assertRaises(TypeError):
            p.preprocess('bla', '*', '*', '*')


if __name__ == "__main__":
    unittest.main()
