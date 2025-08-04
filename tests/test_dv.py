'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    `EUROPEAN UNION PUBLIC LICENCE v. 1.2
    <https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>`_
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 27th October 2021 12:58:15 pm
Last Modified: Monday, 9th December 2024 02:07:22 pm
'''

import unittest
from unittest import mock
from unittest.mock import patch
from copy import deepcopy
import warnings
from zipfile import BadZipFile

import numpy as np
from obspy import UTCDateTime

from seismic.monitor import dv
from seismic.correlate.stats import CorrStats


class TestDV(unittest.TestCase):
    def setUp(self):
        sim_mat = np.reshape(np.arange(500), (10, 50))/500
        corr = np.max(sim_mat, axis=1)
        second_axis = np.random.random(50,)
        value = second_axis[np.argmax(sim_mat, axis=1)]
        self.corr_starts = [
            UTCDateTime(2021, 1, 1, 2*x, 0, 0) for x in range(10)]
        self.corr_ends = [x+2 for x in self.corr_starts]
        self.dv = dv.DV(
            corr, value, 'bla', sim_mat, second_axis, 'blub',
            CorrStats(
                {'corr_start': self.corr_starts, 'corr_end': self.corr_ends}))

    @patch('seismic.monitor.dv.mu.save_header_to_np_array')
    @patch('seismic.monitor.dv.np.savez_compressed')
    def test_save(self, savez_mock, save_header_mock):
        save_header_mock.return_value = {}
        self.dv.save('/save/to/here')
        save_header_mock.assert_called_once_with(self.dv.stats)
        savez_mock.assert_called_once_with(
            '/save/to/here',
            corr=self.dv.corr, value=self.dv.value, sim_mat=self.dv.sim_mat,
            second_axis=self.dv.second_axis,
            method_array=np.array([self.dv.method]),
            vt_array=np.array([self.dv.value_type]))

    @patch('seismic.monitor.dv.mu.save_header_to_np_array')
    @patch('seismic.monitor.dv.np.savez_compressed')
    def test_save2(self, savez_mock, save_header_mock):
        save_header_mock.return_value = {}
        self.dv.corrs = np.random.random((5, 5))
        self.dv.stretches = np.random.random((5, 5))
        self.dv.n_stat = np.ones(5, dtype=int)
        self.dv.dv_processing = dict(
            freq_min=0, freq_max=1, tw_start=2, tw_len=3, sides='bla')
        self.dv.save('/save/to/here')
        save_header_mock.assert_called_once_with(self.dv.stats)
        savez_mock.assert_called_once_with(
            '/save/to/here',
            corr=self.dv.corr, value=self.dv.value, sim_mat=self.dv.sim_mat,
            second_axis=self.dv.second_axis,
            method_array=np.array([self.dv.method]),
            aligned=False,
            vt_array=np.array([self.dv.value_type]), corrs=self.dv.corrs,
            stretches=self.dv.stretches, n_stat=self.dv.n_stat,
            freq_min=0, freq_max=1, tw_start=2, tw_len=3, sides='bla')

    @patch('seismic.monitor.dv.mu.save_header_to_np_array')
    @patch('seismic.monitor.dv.np.savez_compressed')
    def test_save3(self, savez_mock, save_header_mock):
        save_header_mock.return_value = {}
        self.dv.corrs = np.random.random((5, 5))
        self.dv.stretches = np.random.random((5, 5))
        self.dv.n_stat = np.ones(5, dtype=int)
        self.dv.dv_processing = dict(
            freq_min=0, freq_max=1, tw_start=2, tw_len=3)
        self.dv.save('/save/to/here')
        save_header_mock.assert_called_once_with(self.dv.stats)
        savez_mock.assert_called_once_with(
            '/save/to/here',
            corr=self.dv.corr, value=self.dv.value, sim_mat=self.dv.sim_mat,
            second_axis=self.dv.second_axis,
            aligned=False,
            method_array=np.array([self.dv.method]),
            vt_array=np.array([self.dv.value_type]), corrs=self.dv.corrs,
            stretches=self.dv.stretches, n_stat=self.dv.n_stat,
            freq_min=0, freq_max=1, tw_start=2, tw_len=3, sides='unknown')

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

    def test_smooth_thres(self):
        dvc = deepcopy(self.dv)
        dvc.smooth_sim_mat(1, .5)
        ii = self.dv.sim_mat < .5
        np.testing.assert_allclose(dvc.sim_mat[~ii], self.dv.sim_mat[~ii])

    def test_smooth_limit_times(self):
        dvc = deepcopy(self.dv)
        exp = deepcopy(self.dv.value[5])
        dvc.smooth_sim_mat(10, limit_times_to=([9, 59, 0], [10, 5, 0]))
        np.testing.assert_allclose(dvc.value, exp)
    
    def test_smooth_limit_times_exclude(self):
        dvc = deepcopy(self.dv)
        exp = np.nanmean(np.hstack((dvc.value[:4], dvc.value[6:])))
        dvc.smooth_sim_mat(
            10, limit_times_to=([9, 59, 0], [11, 5, 0]),
            limit_times_to_exclude=True)
        np.testing.assert_allclose(dvc.value, exp)


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
            'corrs': None, 'stretches': None, 'n_stat': None,
            'dv_processing': {}, 'avail': True})

    @patch('seismic.monitor.dv.np.load')
    @patch('seismic.monitor.dv.mu.load_header_from_np_array')
    def test2(self, load_header_mock, npload_mock):
        load_header_mock.return_value = {}
        npload_mock.return_value = {
            'corr': np.nan, 'value': 1, 'vt_array': [['s']], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [['d']],
            'stretches': 3, 'corrs': 5, 'freq_min': 0, 'freq_max': 1,
            'tw_start': 2, 'tw_len': 3, 'sides': 'both',
            'aligned': 5}
        dvout = dv.read_dv('/my/dv_file')
        npload_mock.assert_called_once_with('/my/dv_file')
        load_header_mock.assert_called_once_with({
            'corr': np.nan, 'value': 1, 'vt_array': [['s']], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [['d']],
            'corrs': 5, 'stretches': 3, 'freq_min': 0, 'freq_max': 1,
            'tw_start': 2, 'tw_len': 3, 'sides': 'both', 'aligned': 5})
        self.assertDictEqual(dvout.__dict__, {
            'corr': np.nan, 'value': 1, 'value_type': 's', 'sim_mat': 3,
            'second_axis': 4, 'method': 'd', 'stats': CorrStats(),
            'corrs': 5, 'stretches': 3, 'n_stat': None, 'avail': False,
            'dv_processing': {
                'aligned': 5, 'sides': 'both',
                'freq_min': 0., 'freq_max': 1.,
                'tw_start': 2., 'tw_len': 3.}})

    @patch('seismic.monitor.dv.np.load')
    @patch('seismic.monitor.dv.mu.load_header_from_np_array')
    def test3(self, load_header_mock, npload_mock):
        load_header_mock.return_value = {}
        npload_mock.return_value = {
            'corr': 0, 'value': 1, 'vt_array': [['s']], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [['d']],
            'stretches': 3, 'corrs': 5, 'n_stat': 1}
        dvout = dv.read_dv('/my/dv_file')
        npload_mock.assert_called_once_with('/my/dv_file')
        load_header_mock.assert_called_once_with({
            'corr': 0, 'value': 1, 'vt_array': [['s']], 'sim_mat': 3,
            'second_axis': 4, 'method_array': [['d']],
            'corrs': 5, 'stretches': 3, 'n_stat': 1})
        self.assertDictEqual(dvout.__dict__, {
            'corr': 0, 'value': 1, 'value_type': 's', 'sim_mat': 3,
            'second_axis': 4, 'method': 'd', 'stats': CorrStats(),
            'corrs': 5, 'stretches': 3, 'n_stat': 1, 'dv_processing': {},
            'avail': True})

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
            'corrs': None, 'stretches': None, 'n_stat': None,
            'dv_processing': {}, 'avail': True})
        self.assertDictEqual(dvout[1].__dict__, {
            'corr': 1, 'value': 2, 'value_type': '3', 'sim_mat': 4,
            'second_axis': 5, 'method': 'd', 'stats': CorrStats(),
            'corrs': None, 'stretches': None, 'n_stat': None,
            'dv_processing': {}, 'avail': True})
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
            'corrs': None, 'stretches': None, 'n_stat': None,
            'dv_processing': {}, 'avail': True})
        self.assertEqual(len(dvout), 1)


if __name__ == "__main__":
    unittest.main()
