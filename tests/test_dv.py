'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 27th October 2021 12:58:15 pm
Last Modified: Wednesday, 27th October 2021 01:41:58 pm
'''

import unittest
from unittest.mock import patch
from copy import deepcopy

import numpy as np

from seismic.monitor import dv
from seismic.correlate.stats import CorrStats


class TestDV(unittest.TestCase):
    def setUp(self):
        sim_mat = np.reshape(np.arange(500), (10, 50))
        corr = np.max(sim_mat, axis=1)
        second_axis = np.random.random(50,)
        value = second_axis[np.argmax(sim_mat, axis=1)]
        self.dv = dv.DV(corr, value, 'bla', sim_mat, second_axis, 'blub', {})

    @patch('seismic.monitor.dv.save_header_to_np_array')
    @patch('seismic.monitor.dv.np.savez_compressed')
    def test_save(self, savez_mock, save_header_mock):
        self.dv.save('/save/to/here')
        save_header_mock.assert_called_once_with({})
        savez_mock.assert_called_once_with(
            '/save/to/here',
            corr=self.dv.corr, value=self.dv.value, sim_mat=self.dv.sim_mat,
            second_axis=self.dv.second_axis,
            method_array=np.array([self.dv.method]),
            vt_array=np.array([self.dv.value_type]))

    def test_smooth_sim_mat(self):
        dvc = deepcopy(self.dv)
        dvc.smooth_sim_mat(5)
        self.assertAlmostEqual(dvc.sim_mat.sum(), self.dv.sim_mat.sum())

    def test_smooth_win1(self):
        dvc = deepcopy(self.dv)
        dvc.smooth_sim_mat(1)
        np.testing.assert_allclose(dvc.sim_mat, self.dv.sim_mat)
        np.testing.assert_allclose(dvc.value, self.dv.value)
        np.testing.assert_allclose(dvc.corr, self.dv.corr)


class TestReadDV(unittest.TestCase):
    @patch('seismic.monitor.dv.np.load')
    @patch('seismic.monitor.dv.load_header_from_np_array')
    def test(self, load_header_mock, npload_mock):
        load_header_mock.return_value = {}
        npload_mock.return_value = {
            'corr': 0, 'value': 1, 'vt_array': [2], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [5]}
        dvout = dv.read_dv('/my/dv_file')
        npload_mock.assert_called_once_with('/my/dv_file')
        load_header_mock.assert_called_once_with({
            'corr': 0, 'value': 1, 'vt_array': [2], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [5]})
        self.assertDictEqual(dvout.__dict__, {
            'corr': 0, 'value': 1, 'value_type': 2, 'sim_mat': 3,
            'second_axis': 4, 'method': 5, 'stats': CorrStats()})


if __name__ == "__main__":
    unittest.main()