'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 7th July 2023 02:50:27 pm
Last Modified: Wednesday, 19th July 2023 11:58:20 am
'''

import unittest
from unittest import mock

import numpy as np
from obspy import read

import seismic.utils.raw_analysis as ra


def side_effect_func(value1, value2):
    return value1


class TestSpctSeriesWelch(unittest.TestCase):
    """
    Tests for spct_series_welch.
    """

    def setUp(self):
        """
        Set up test data.
        """
        self.st = read()
        for tr in self.st:
            tr.stats.sampling_rate = 1
            tr.data = np.random.rand(100)

    @mock.patch('seismic.utils.raw_analysis.preprocess')
    def test_spct_series_welch(self, mock_preprocess):
        """
        Test spct_series_welch.
        """
        mock_preprocess.side_effect = side_effect_func
        f, t, S = ra.spct_series_welch([self.st], 10, 50)
        self.assertEqual(f.shape, (512,))
        self.assertEqual(t.shape, (9,))
        self.assertEqual(S.shape, (512, 9))
        mock_preprocess.assert_has_calls(
            [mock.call(mock.ANY, 50) for tr in self.st.split()])


class TestPreprocess(unittest.TestCase):
    """
    Tests for preprocess.
    """

    def setUp(self):
        """
        Set up test data.
        """
        self.tr = mock.MagicMock()

    @mock.patch('seismic.utils.raw_analysis.resample_or_decimate')
    def test_preprocess(self, mock_resample_or_decimate):
        """
        Test preprocess.
        """
        tr = ra.preprocess(self.tr, 25)
        tr.detrend.assert_called_once_with(type='linear')
        mock_resample_or_decimate.assert_called_once_with(self.tr, 50)
        tr.filter.assert_called_once_with('highpass', freq=0.01)
        tr.remove_response.assert_called_once()


if __name__ == '__main__':
    unittest.main()
