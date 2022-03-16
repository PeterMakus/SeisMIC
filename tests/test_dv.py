'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 27th October 2021 12:58:15 pm
Last Modified: Wednesday, 16th March 2022 01:25:42 pm
'''

import unittest
from unittest import mock
from unittest.mock import patch
from copy import deepcopy
import warnings
from zipfile import BadZipFile

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

    @patch('seismic.monitor.dv.mu.save_header_to_np_array')
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

    @patch('seismic.monitor.dv.mu.save_header_to_np_array')
    @patch('seismic.monitor.dv.np.savez_compressed')
    def test_save2(self, savez_mock, save_header_mock):
        self.dv.std_corr = np.random.random(5)
        self.dv.std_val = np.random.random(5)
        self.dv.n_stat = np.ones(5, dtype=int)
        self.dv.save('/save/to/here')
        save_header_mock.assert_called_once_with({})
        savez_mock.assert_called_once_with(
            '/save/to/here',
            corr=self.dv.corr, value=self.dv.value, sim_mat=self.dv.sim_mat,
            second_axis=self.dv.second_axis,
            method_array=np.array([self.dv.method]),
            vt_array=np.array([self.dv.value_type]), std_corr=self.dv.std_corr,
            std_val=self.dv.std_val, n_stat=self.dv.n_stat)

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
    @patch('seismic.monitor.dv.mu.load_header_from_np_array')
    def test(self, load_header_mock, npload_mock):
        load_header_mock.return_value = {}
        npload_mock.return_value = {
            'corr': 0, 'value': 1, 'vt_array': [['s']], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [['d']]}
        dvout = dv.read_dv('/my/dv_file')
        npload_mock.assert_called_once_with('/my/dv_file')
        load_header_mock.assert_called_once_with({
            'corr': 0, 'value': 1, 'vt_array': [['s']], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [['d']]})
        self.assertDictEqual(dvout.__dict__, {
            'corr': 0, 'value': 1, 'value_type': 's', 'sim_mat': 3,
            'second_axis': 4, 'method': 'd', 'stats': CorrStats(),
            'std_corr': None, 'std_val': None, 'n_stat': None})

    @patch('seismic.monitor.dv.np.load')
    @patch('seismic.monitor.dv.mu.load_header_from_np_array')
    def test2(self, load_header_mock, npload_mock):
        load_header_mock.return_value = {}
        npload_mock.return_value = {
            'corr': 0, 'value': 1, 'vt_array': [['s']], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [['d']],
            'std_val': 3, 'std_corr': 5}
        dvout = dv.read_dv('/my/dv_file')
        npload_mock.assert_called_once_with('/my/dv_file')
        load_header_mock.assert_called_once_with({
            'corr': 0, 'value': 1, 'vt_array': [['s']], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [['d']],
            'std_corr': 5, 'std_val': 3})
        self.assertDictEqual(dvout.__dict__, {
            'corr': 0, 'value': 1, 'value_type': 's', 'sim_mat': 3,
            'second_axis': 4, 'method': 'd', 'stats': CorrStats(),
            'std_corr': 5, 'std_val': 3, 'n_stat': None})

    @patch('seismic.monitor.dv.np.load')
    @patch('seismic.monitor.dv.mu.load_header_from_np_array')
    def test3(self, load_header_mock, npload_mock):
        load_header_mock.return_value = {}
        npload_mock.return_value = {
            'corr': 0, 'value': 1, 'vt_array': [['s']], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [['d']],
            'std_val': 3, 'std_corr': 5, 'n_stat': 1}
        dvout = dv.read_dv('/my/dv_file')
        npload_mock.assert_called_once_with('/my/dv_file')
        load_header_mock.assert_called_once_with({
            'corr': 0, 'value': 1, 'vt_array': [['s']], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [['d']],
            'std_corr': 5, 'std_val': 3, 'n_stat': 1})
        self.assertDictEqual(dvout.__dict__, {
            'corr': 0, 'value': 1, 'value_type': 's', 'sim_mat': 3,
            'second_axis': 4, 'method': 'd', 'stats': CorrStats(),
            'std_corr': 5, 'std_val': 3, 'n_stat': 1})

    @patch('seismic.monitor.dv.glob')
    @patch('seismic.monitor.dv.np.load')
    @patch('seismic.monitor.dv.mu.load_header_from_np_array')
    def test_pattern(
        self, load_header_mock: mock.MagicMock, npload_mock: mock.MagicMock,
            glob_mock: mock.MagicMock):
        load_header_mock.return_value = {}
        npload_mock.side_effect = ({
            'corr': 0, 'value': 1, 'vt_array': [[['b']]], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [[['xs']]]},
            {
            'corr': 1, 'value': 2, 'vt_array': [['3']], 'sim_mat': 4,
            'second_axis': 5, 'method_array': [['d']]})
        glob_mock.return_value = ['/my/dv0', '/my/dv1']
        dvout = dv.read_dv('/my/dv?')
        npload_calls = [mock.call('/my/dv0'), mock.call('/my/dv1')]
        npload_mock.assert_has_calls(npload_calls)
        lheader_calls = [mock.call({
            'corr': 0, 'value': 1, 'vt_array': [[['b']]], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [[['xs']]]}),
            mock.call({
                'corr': 1, 'value': 2, 'vt_array': [['3']], 'sim_mat': 4,
                'second_axis': 5, 'method_array': [['d']]})]
        load_header_mock.assert_has_calls(lheader_calls)
        self.assertDictEqual(dvout[0].__dict__, {
            'corr': 0, 'value': 1, 'value_type': 'b', 'sim_mat': 3,
            'second_axis': 4, 'method': 'xs', 'stats': CorrStats(),
            'std_corr': None, 'std_val': None, 'n_stat': None})
        self.assertDictEqual(dvout[1].__dict__, {
            'corr': 1, 'value': 2, 'value_type': '3', 'sim_mat': 4,
            'second_axis': 5, 'method': 'd', 'stats': CorrStats(),
            'std_corr': None, 'std_val': None, 'n_stat': None})
        self.assertEqual(len(dvout), 2)

    @patch('seismic.monitor.dv.glob')
    def test_pattern_not_found(self, glob_mock: mock.MagicMock):
        glob_mock.return_value = []
        with self.assertRaises(FileNotFoundError):
            dv.read_dv('/my/dv*')

    @patch('seismic.monitor.dv.glob')
    @patch('seismic.monitor.dv.np.load')
    @patch('seismic.monitor.dv.mu.load_header_from_np_array')
    def test_badzipfile(
        self, load_header_mock: mock.MagicMock, npload_mock: mock.MagicMock,
            glob_mock: mock.MagicMock):
        load_header_mock.return_value = {}
        npload_mock.side_effect = ({
            'corr': 0, 'value': 1, 'vt_array': [[['b']]], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [[['xs']]]},
            BadZipFile)
        glob_mock.return_value = ['/my/dv0', '/my/dv1']
        with warnings.catch_warnings(record=True) as w:
            dvout = dv.read_dv('/my/dv?')
            self.assertEqual(len(w), 1)
        npload_calls = [mock.call('/my/dv0'), mock.call('/my/dv1')]
        npload_mock.assert_has_calls(npload_calls)
        lheader_calls = [mock.call({
            'corr': 0, 'value': 1, 'vt_array': [[['b']]], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [[['xs']]]})]
        load_header_mock.assert_has_calls(lheader_calls)
        self.assertDictEqual(dvout[0].__dict__, {
            'corr': 0, 'value': 1, 'value_type': 'b', 'sim_mat': 3,
            'second_axis': 4, 'method': 'xs', 'stats': CorrStats(),
            'std_corr': None, 'std_val': None, 'n_stat': None})
        self.assertEqual(len(dvout), 1)


if __name__ == "__main__":
    unittest.main()
