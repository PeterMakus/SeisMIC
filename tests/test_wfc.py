'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 7th July 2023 03:16:14 pm
Last Modified: Friday, 7th July 2023 04:07:39 pm
'''

import unittest
from unittest import mock

import numpy as np

import seismic.monitor.wfc as wfc


class TestWFC(unittest.TestCase):
    def setUp(self):
        self.wfc_dict = {
            'reftr_0': np.random.rand(100),
            'reftr_1': np.random.rand(100)+np.nan,
            'reftr_2': np.random.rand(100),
        }
        self.wfc_processing = {
            'freq_min': 1,
            'freq_max': 10,
            'tw_start': 0,
            'tw_len': 10}
        self.stats = mock.MagicMock()
        self.wc = wfc.WFC(self.wfc_dict, self.wfc_processing, self.stats)

    def test_compute_average(self):
        self.wc.compute_average()
        self.assertAlmostEqual(
            self.wc.mean,
            np.mean([self.wfc_dict['reftr_0'], self.wfc_dict['reftr_2']]))

    @mock.patch('seismic.monitor.wfc.np.savez_compressed')
    def test_save(self, mock_savez_compressed):
        self.wc.save('test')
        mock_savez_compressed.assert_called_once_with(
            'test',
            mean=self.wc.mean,
            reftr_0=self.wfc_dict['reftr_0'],
            reftr_1=self.wfc_dict['reftr_1'],
            reftr_2=self.wfc_dict['reftr_2'],
            **self.wfc_processing)


class TestWFCBulk(unittest.TestCase):
    def setUp(self):
        self.wfcs = []
        for i in range(10):
            wfc_dict = {
                'reftr_0': np.random.rand(100),
            }
            wfc_processing = {
                'freq_min': i,
                'freq_max': i*2,
                'tw_start': i,
                'tw_len': 1}
            stats = mock.MagicMock()
            self.wfcs.append(
                wfc.WFC(wfc_dict, stats, wfc_processing))
        self.wfcs[-1].mean = np.array(1)
        self.wfc_bulk = wfc.WFCBulk(self.wfcs)

    def test_init(self):
        self.assertEqual(self.wfc_bulk.wfc.shape, (10, 10))
        np.testing.assert_array_equal(
            self.wfc_bulk.cfreq, 1.5*np.arange(10))
        np.testing.assert_array_equal(
            self.wfc_bulk.lw, np.arange(10) + .5)

    @mock.patch('seismic.monitor.wfc.plot_wfc_bulk')
    def test_plot(self, mock_plot_wfc_bulk):
        self.wfc_bulk.plot()
        mock_plot_wfc_bulk.assert_called_once_with(
            self.wfc_bulk.lw, self.wfc_bulk.cfreq, self.wfc_bulk.wfc,
            title=None, log=False, cmap='viridis')


if __name__ == '__main__':
    unittest.main()
