'''
:copyright:
:license:
    `EUROPEAN UNION PUBLIC LICENCE v. 1.2
    <https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>`_
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 27th May 2021 04:27:14 pm
Last Modified: Monday, 04th December 2024 03:07:26 pm (J. Lehr)
'''
from copy import deepcopy
import unittest
import warnings
from unittest import mock
import os

from mpi4py import MPI
import numpy as np
from obspy import read, Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.core.inventory.inventory import read_inventory, Inventory
import yaml

from seismic.correlate import correlate
from seismic.correlate.stream import CorrStream
from seismic.trace_data.waveform import Store_Client

paramfile = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'params_example.yaml')


@mock.patch('builtins.open')
@mock.patch('seismic.correlate.correlate.logfactory.LoggingMPIBaseClass')
@mock.patch('seismic.correlate.correlate.os.makedirs')
class TestCorrrelator(unittest.TestCase):
    mock_logger = mock.create_autospec(correlate.logging.Logger)

    def setUp(self):
        self.inv = read_inventory()
        self.st = read()
        # using the example parameters has the nice side effect that
        # the parameter file is tested as well
        self.param_example = paramfile
        with open(self.param_example) as file:
            self.options = yaml.load(file, Loader=yaml.FullLoader)
        self.options['co']['preprocess_subdiv'] = True

    @mock.patch('seismic.correlate.correlate.yaml.load')
    def test_filter_by_rcombis(self, yaml_mock, makedirs_mock, logging_mock,
                               open_mock):
        yaml_mock.return_value = self.options
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        sc_mock._translate_wildcards.return_value = []
        c = correlate.Correlator(self.param_example, sc_mock)
        c.station = [
            ['NET1', 'STA1'], ['NET1', 'STA2'], ['NET2', 'STA1']]
        c.avail_raw_data = [
            ['NET1', 'STA1', 'LOC1', 'CHAN1'],
            ['NET1', 'STA2', 'LOC2', 'CHAN1'],
            ['NET2', 'STA1', 'LOC1', 'CHAN1']]
        c.rcombis = ['NET1-NET2.STA1-STA1']
        c._filter_by_rcombis()
        self.assertListEqual(
            c.station, [['NET1', 'STA1'], ['NET2', 'STA1']])
        self.assertListEqual(
            c.avail_raw_data, [
                ['NET1', 'STA1', 'LOC1', 'CHAN1'],
                ['NET2', 'STA1', 'LOC1', 'CHAN1']])

    @mock.patch('seismic.correlate.correlate.yaml.load')
    def test_init_options_from_yaml(
            self, yaml_mock, makedirs_mock, logging_mock, open_mock):
        yaml_mock.return_value = self.options
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        sc_mock._translate_wildcards.return_value = []
        c = correlate.Correlator(self.param_example, sc_mock)
        self.assertDictEqual(self.options['co'], c.options)
        mkdir_calls = [
            mock.call(os.path.join(
                self.options['proj_dir'], self.options['log_subdir']),
                exist_ok=True),
            mock.call(os.path.join(
                self.options['proj_dir'], self.options['co']['subdir']),
                exist_ok=True),
            ]
        makedirs_mock.assert_has_calls(mkdir_calls)
        open_mock.assert_has_calls([mock.call(file=self.param_example)])

    @mock.patch('seismic.correlate.correlate.yaml.load')
    def test_deprecation_of_args(
            self, yaml_mock, makedirs_mock, logging_mock, open_mock):
        yaml_mock.return_value = self.options
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        sc_mock._translate_wildcards.return_value = []
        # c = correlate.Correlator(sc_mock, self.param_example)
        self.assertRaises(DeprecationWarning,
                          correlate.Correlator, sc_mock, self.param_example)

    @mock.patch('seismic.correlate.correlate.yaml.load')
    @mock.patch('seismic.trace_data.waveform.os.listdir')
    @mock.patch('seismic.trace_data.waveform.os.path.isdir')
    @mock.patch('seismic.trace_data.waveform.read_inventory')
    @mock.patch('obspy.clients.filesystem.sds.os.path.isdir')
    def test_init_without_storeclient(
            self, sds_exists_mock, read_inventory_mock,
            isdir_mock, listdir_mock, yaml_mock, makedirs_mock,
            logging_mock, open_mock):

        yaml_mock.return_value = self.options
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        sc_mock._translate_wildcards.return_value = []
        isdir_mock.return_value = True
        listdir_mock.return_value = False
        read_inventory_mock.return_value = Inventory()
        sds_exists_mock.return_value = True
        c = correlate.Correlator(self.param_example)
        self.assertDictEqual(self.options['co'], c.options)
        mkdir_calls = [
            mock.call(os.path.join(
                self.options['proj_dir'], self.options['log_subdir']),
                exist_ok=True),
            mock.call(os.path.join(
                self.options['proj_dir'], self.options['co']['subdir']),
                exist_ok=True),
            mock.call(os.path.join(
                self.options['proj_dir']), exist_ok=True),
            mock.call(os.path.join(
                self.options['proj_dir'], self.options['log_subdir']),
                exist_ok=True)]
        makedirs_mock.assert_has_calls(mkdir_calls)
        open_mock.assert_has_calls([mock.call(file=self.param_example)])

    def test_stat_net_str(
            self, makedirs_mock, logging_mock, open_mock):
        # Test the change of UTCDateTime object
        self.options['co']['preProcessing'].append(
            {
                'function': '...stream_mask_at_utc',
                'args': {'starts': [UTCDateTime(0)], 'ends': [
                    UTCDateTime(10)]}})
        options = deepcopy(self.options)
        options['net']['network'] = ['bla']
        options['net']['station'] = ['blub']
        options['net']['component'] = 'E'
        sc_mock = mock.Mock(Store_Client)
        sc_mock._translate_wildcards.return_value = [
            ['bla', 'blub', '00', 'E']]
        c = correlate.Correlator(options, sc_mock)
        sc_mock._translate_wildcards.assert_called_once_with(
            'bla', 'blub', 'E', location='*')
        sc_mock.get_available_stations.assert_not_called()
        # What is the purpose of this test? It does not work anymore with
        # the joint_norm argument because it changes the function definition
        # in the options dict.
        # self.assertDictEqual(self.options['co'], c.options)
        self.assertListEqual(c.station, [['bla', 'blub']])

    def test_net_wc_stat_str(
            self, makedirs_mock, logging_mock, open_mock):
        # Test the change of UTCDateTime object
        self.options['co']['preProcessing'].append(
            {
                'function': '...stream_mask_at_utc',
                'args': {'starts': [UTCDateTime(0)], 'masklen': 10}})
        options = deepcopy(self.options)
        options['net']['network'] = '*'
        options['net']['station'] = ['blub']
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        with self.assertRaises(ValueError):
            correlate.Correlator(options, sc_mock)

    def test_net_wc(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = '*'
        options['net']['station'] = '*'
        sc_mock = mock.Mock(Store_Client)
        sc_mock._translate_wildcards.return_value = [['lala', 'lolo', 'E']]
        c = correlate.Correlator(options, sc_mock)
        self.assertListEqual(c.station, [['lala', 'lolo']])

    def test_stat_wc(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = 'lala'
        options['net']['station'] = '*'
        options['net']['component'] = None
        sc_mock = mock.Mock(Store_Client)
        sc_mock._translate_wildcards.side_effect = (
            [['lala', 'lolo', '00', 'E']], [['lala', 'lolo', '11', 'Z']])
        c = correlate.Correlator(options, sc_mock)
        self.assertListEqual(c.station, [['lala', 'lolo']])
        sc_mock._translate_wildcards.assert_called_once_with(
            'lala', '*', None, location='*')

    def test_stat_wc_net_list(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = ['lala', 'lolo']
        options['net']['station'] = '*'
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [['lala', 'lolo']]
        sc_mock._translate_wildcards.side_effect = (
            [['lala', 'lala', 'E']],
            [['lolo', 'lolo', 'E']])
        c = correlate.Correlator(options, sc_mock)
        self.assertListEqual(
            c.station, [['lala', 'lala'], ['lolo', 'lolo']])
        calls = [mock.call('lala'), mock.call('lolo')]
        sc_mock.get_available_stations.assert_has_calls(calls)

    def test_list_len_diff(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = [1, 2, 3]
        options['net']['station'] = ['blub', 'blib']
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        with self.assertRaises(ValueError):
            correlate.Correlator(options, sc_mock)

    def test_stat_net_list(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = ['lala', 'lolo']
        options['net']['station'] = ['le', 'li']
        sc_mock = mock.Mock(Store_Client)
        sc_mock._translate_wildcards.side_effect = (
            [['lala', 'le', '00', 'E']], [['lolo', 'li', '01', 'Z']])
        c = correlate.Correlator(options, sc_mock)
        sc_mock._translate_wildcards.assert_has_calls([
            mock.call('lala', 'le', 'Z', location='*'),
            mock.call('lolo', 'li', 'Z', location='*')])
        sc_mock.get_available_stations.assert_not_called()
        self.assertListEqual(
            c.station, [['lala', 'le'], ['lolo', 'li']])

    def test_other_wrong(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = 1
        options['net']['station'] = ['blub']
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        with self.assertRaises(ValueError):
            correlate.Correlator(options, sc_mock)

    @mock.patch('seismic.correlate.correlate.mu.filter_stat_dist')
    def test_find_interstat_dis(
            self, statd_mock, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = '*'
        options['net']['station'] = '*'
        options['co']['combination_method'] = 'betweenStations'
        sc_mock = mock.Mock(Store_Client)
        sc_mock._translate_wildcards.return_value = [
            ['lala', 'lolo', 'E'], ['lala', 'lili', 'Z']]
        c = correlate.Correlator(options, sc_mock)

        sc_mock.select_inventory_or_load_remote.side_effect = [
            'a', 'b', 'c', 'd', 'e', 'f']
        statd_mock.return_value = True
        c.find_interstat_dist(10000)
        sc_mock.read_inventory.assert_called_once()
        sc_mock.select_inventory_or_load_remote.assert_called_with(
            'lala', 'lolo')
        self.assertSetEqual(
            set(c.rcombis),
            {
                'lala-lala.lili-lolo',
                'lala-lala.lili-lili', 'lala-lala.lolo-lolo'})

    def test_find_interstat_dis_wrong_method(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = '*'
        options['net']['station'] = '*'
        options['co']['combination_method'] = 'betweenComponents'
        sc_mock = mock.Mock(Store_Client)
        sc_mock._translate_wildcards.return_value = [
            ['lala', 'lolo', 'E'], ['lala', 'lili', 'Z']]
        c = correlate.Correlator(options, sc_mock)
        with self.assertRaises(ValueError):
            c.find_interstat_dist(100)

    @mock.patch('seismic.db.corr_hdf5.DBHandler')
    @mock.patch('seismic.correlate.correlate.glob.glob')
    @mock.patch(
        'seismic.correlate.correlate.compute_network_station_combinations')
    def test_find_existing_times(
        self, ccomb_mock, isfile_mock,
            cdb_mock, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [
            ['lala', 'lolo'], ['lala', 'lili']]
        sc_mock._translate_wildcards.return_value = [
            ['lala', 'lolo', '00', 'E'], ['lala', 'lili', '01', 'Z']]
        c = correlate.Correlator(options, sc_mock)
        netcombs = ['AA-BB', 'AA-BB', 'AA-AA', 'AA-CC']
        statcombs = ['00-00', '00-11', '22-33', '22-44']
        ccomb_mock.return_value = (netcombs, statcombs)
        isfile = [
            [1], ['AA-BB.00-00.00-01.E-Z.h5'], [], [2],
            ['AA-AA.22-33.00-01.E-Z.h5'],
            [], [], []]
        isfile_mock.side_effect = isfile
        times = [{'a': [0, 1, 2]}, {'b': [3, 4, 5, 6]}]
        cdb_mock().get_available_starttimes.side_effect = times
        out = c.find_existing_times('mytag')
        exp = {
            'AA.00': {'BB.00': {'00-01': {'a': [0, 1, 2]}}},
            'AA.22': {'AA.33': {'00-01': {'b': [3, 4, 5, 6]}}}}
        self.assertDictEqual(out, exp)
        # isfile_calls = [
        #     os.path.join(c.corr_dir, f'{nc}.{sc}*.h5') for nc, sc in zip(
        #         netcombs, statcombs)]
        # for call in isfile_calls:
        #     isfile_mock.assert_any_call(call)

    @mock.patch('seismic.correlate.correlate.CorrStream')
    def test_pxcorr(self, cst_mock, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['co']['subdivision']['recombine_subdivision'] = True
        options['co']['subdivision']['delete_subdivision'] = True
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [
            ['lala', 'lolo'], ['lala', 'lili']]
        sc_mock._translate_wildcards.return_value = [
            ['lala', 'lolo', 'E'], ['lala', 'lili', 'Z']]
        c = correlate.Correlator(options, sc_mock)
        sc_mock.read_inventory.return_value = self.inv
        cst_mock().stack.return_value = 'bla'
        cst_mock().count.return_value = True
        with mock.patch.multiple(
            c, _generate_data=mock.MagicMock(return_value=[[self.st, True]]),
            _pxcorr_inner=mock.MagicMock(return_value=self.st),
                _write=mock.MagicMock()):
            c.pxcorr()
            c._generate_data.assert_called_once()
            c._pxcorr_inner.assert_called_once_with(self.st, self.inv)
            cst_mock().count.assert_called()
            write_calls = [mock.call(cst_mock()), mock.call(cst_mock())]
            c._write.assert_has_calls(write_calls)
        cst_mock().clear.assert_called()
        cst_mock().extend.assert_called_once()

    @mock.patch('seismic.correlate.correlate.CorrStream')
    def test_pxcorr2(self, cst_mock, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['co']['subdivision']['recombine_subdivision'] = False
        options['co']['subdivision']['delete_subdivision'] = False
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [
            ['lala', 'lolo'], ['lala', 'lili']]
        sc_mock._translate_wildcards.return_value = [
            ['lala', 'lolo', 'E'], ['lala', 'lili', 'Z']]
        c = correlate.Correlator(options, sc_mock)
        sc_mock.read_inventory.return_value = self.inv
        cst_mock().count.return_value = True
        with mock.patch.multiple(
            c, _generate_data=mock.MagicMock(return_value=[[self.st, True]]),
            _pxcorr_inner=mock.MagicMock(return_value=self.st),
                _write=mock.MagicMock()):
            c.pxcorr()
            c._generate_data.assert_called_once()
            c._pxcorr_inner.assert_called_once_with(self.st, self.inv)
            cst_mock().stack.assert_not_called()
            cst_mock().count.assert_called()
            write_calls = [
                mock.call(mock.ANY),
                mock.call(mock.ANY)
            ]
            c._write.assert_has_calls(write_calls)
        cst_mock().extend.assert_called_once()

    @mock.patch('seismic.correlate.correlate.st_to_np_array')
    def test_pxcorr_inner(
            self, st_a_mock, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['co']['combinations'] = [(0, 0), (0, 1), (0, 2)]
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [
            ['lala', 'lolo'], ['lala', 'lili']]
        sc_mock._translate_wildcards.return_value = [
            ['lala', 'lolo', 'E'], ['lala', 'lili', 'Z']]
        c = correlate.Correlator(options, sc_mock)
        st_a_mock.return_value = (np.zeros((3, 5)), self.st)
        with mock.patch.object(c, '_pxcorr_matrix') as pxcm:
            pxcm.return_value = (np.ones((3, 5)), np.arange(5))
            cst = c._pxcorr_inner(self.st, self.inv)
            pxcm.assert_called_once()
        self.assertListEqual(
            c.options['starttime'], [tr.stats.starttime for tr in self.st])
        for ctr in cst:
            np.testing.assert_array_equal(np.ones(5,), ctr.data)
        self.assertEqual(cst.count(), 3)

    @mock.patch('seismic.db.corr_hdf5.DBHandler')
    def test_write_three_file(
            self, dbh_mock, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['co']['combinations'] = [(0, 0), (0, 1), (0, 2)]
        options['co']['subdivision']['recombine_subdivision'] = True
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [
            ['lala', 'lolo'], ['lala', 'lili']]
        sc_mock._translate_wildcards.return_value = [
            ['lala', 'lolo', 'E'], ['lala', 'lili', 'Z']]
        c = correlate.Correlator(options, sc_mock)
        cst_mock = mock.Mock(CorrStream)
        cst_mock2 = mock.Mock(CorrStream)
        cst_mock.select.return_value = cst_mock2
        # make cst_mock iterable
        cst_mock.__iter__ = mock.Mock(return_value=iter(self.st))
        c._write(cst_mock)
        select_calls = [
            mock.call(
                network='BW', station='RJOB', location='', channel='EHE'),
            mock.call().stack(),
            mock.call().count(),
            mock.call(
                network='BW', station='RJOB', location='', channel='EHN'),
            mock.call().stack(),
            mock.call().count(),
            mock.call(
                network='BW', station='RJOB', location='', channel='EHZ'),
            mock.call().stack(),
            mock.call().count()]
        cst_mock.select.assert_has_calls(select_calls)
        # Note that subdivision is still called because clear doesn't
        # have a defined effect on a mock
        add_cst_calls = [
            mock.call(mock.ANY, 'subdivision'),
            mock.call(mock.ANY, 'stack_86398'),
            mock.call(mock.ANY, 'subdivision'),
            mock.call(mock.ANY, 'stack_86398'),
            mock.call(mock.ANY, 'subdivision'),
            mock.call(mock.ANY, 'stack_86398')]
        dbh_mock().add_correlation.assert_has_calls(add_cst_calls)

    @mock.patch('seismic.utils.miic_utils.resample_or_decimate')
    @mock.patch('seismic.correlate.correlate.calc_cross_combis')
    @mock.patch('seismic.correlate.correlate.preprocess_stream')
    @mock.patch('seismic.correlate.correlate.mu.get_valid_traces')
    def test_generate(
        self, gvt_mock, ppst_mock, ccc_mock, rod_mock,
            makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['co']['subdivision']['corr_inc'] = 5
        options['co']['subdivision']['corr_len'] = 5
        ccc_mock.return_value = [(0, 0), (0, 1), (0, 2)]
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [
            ['lala', 'lolo'], ['lala', 'lili']]
        sc_mock._translate_wildcards.return_value = [
            ['lala', 'lolo', '00', 'E'], ['lala', 'lili', '01', 'Z']]
        sc_mock.inventory = self.inv
        sc_mock._load_local.return_value = self.st
        ppst_mock.return_value = self.st
        rod_mock.return_value = self.st
        c = correlate.Correlator(options, sc_mock)
        c.station = [['AA', '00'], ['AA', '22'], ['AA', '33'], ['BB', '00']]
        ostart = None
        for win, write_flag in c._generate_data():
            if ostart and not write_flag:
                # Also, good way to check whether the write_flag comes at
                # the right place
                self.assertAlmostEqual(win[0].stats.starttime - ostart, 5)
            ostart = win[0].stats.starttime
            # The last one could be shorter
            self.assertAlmostEqual(
                win[0].stats.endtime-win[0].stats.starttime, 5, 1)

    @mock.patch.object(correlate.Correlator,
                       "_get_row_index_per_core")
    @mock.patch('seismic.correlate.correlate.pptd.zeroPadding')
    @mock.patch('seismic.correlate.correlate.func_from_str')
    def test_pxcorr_matrix(
            self, ffs_mock, zp_mock, ri_mock,
            makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['co']['combinations'] = [(0, 0), (0, 1), (0, 2)]
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [
            ['lala', 'lolo'], ['lala', 'lili']]
        sc_mock._translate_wildcards.return_value = [
            ['lala', 'lolo', 'E'], ['lala', 'lili', 'Z']]
        sc_mock.select_inventory_or_load_remote.return_value = self.inv
        sc_mock._load_local.return_value = self.st
        c = correlate.Correlator(options, sc_mock)
        c.options['corr_args']['FDpreProcessing'] = [
            {'function': 'FDPP', 'args': []}]
        c.options['corr_args']['TDpreProcessing'] = [
            {'function': 'TDPP', 'args': []}]
        c.options['corr_args']['lengthToSave'] = 1
        c.options.update(
            {'starttime': [tr.stats.starttime for tr in self.st],
                'sampling_rate': self.st[0].stats.sampling_rate})
        # 13 is the fft size
        shape = (25, 101)
        ftshape = (25, 51)
        ri_mock.return_value = np.ones(shape[0], dtype=bool)
        return_func = mock.MagicMock(side_effect=[
            np.ones(shape, dtype=np.float32),
            np.ones(ftshape, dtype=np.float32)])
        ffs_mock.return_value = return_func
        zp_mock.return_value = np.ones(shape)*2
        C, startlags = c._pxcorr_matrix(np.zeros(shape))
        ri_mock.assert_called_once()
        ffs_mock.assert_any_call('FDPP')
        ffs_mock.assert_any_call('TDPP')
        # This is the same as asserthascalls, but it can also check np arrays
        np.testing.assert_array_equal(
            np.zeros(shape), return_func.call_args_list[0][0][0])
        self.assertListEqual([], return_func.call_args_list[0][0][1])
        # Check if the fft worked
        np.testing.assert_array_almost_equal(
            np.fft.rfft(np.ones(shape)*2),
            return_func.call_args_list[1][0][0])
        np.testing.assert_array_equal(-np.ones((3,)), startlags)
        # Correlation should be one in the middle
        # Length is 51,3 as above
        expC = np.zeros((3, 51))
        expC[:, 25] += 1
        np.testing.assert_array_almost_equal(C, expC, decimal=2)

    @mock.patch.object(correlate.Correlator, '__init__')
    def test_set_joint_norm_args(self, mock_init, makedirs_mock, logging_mock,
                                 open_mock):
        mock_init.return_value = None
        options = deepcopy(self.options)
        co_options_new = deepcopy(self.options["co"])
        co_options_new['corr_args']['FDpreProcessing'][0]['args'] = {
            'joint_norm': False}
        # sc_mock = mock.Mock(Store_Client)
        # sc_mock.get_available_stations.return_value = []
        # sc_mock._translate_wildcards.return_value = []
        c = correlate.Correlator()

        mock_comm = mock.Mock()
        c.logger = logging_mock
        c.rank = 0
        c.comm = mock_comm
        c.options = options["co"]
        mock_comm.bcast.return_value = c.options
        c._set_joint_norm_arg()
        self.assertDictEqual(c.options, co_options_new)

    @mock.patch.object(correlate.Correlator, '__init__')
    def test_check_joint_norm_atleast3comps(
            self, mock_init, makedirs_mock, logging_mock, open_mock):
        mock_init.return_value = None
        joint_norm = True
        co_options = deepcopy(self.options["co"])
        co_options['joint_norm'] = joint_norm
        co_options['corr_args']['FDpreProcessing'][0]['args'] = {
            'joint_norm': joint_norm}

        avail_raw_data = sorted(
            [["NET", "STA%s" % i, "", "HH%s" % c]
                for i in "1" for c in "ZN"])
        avail_stations = [item[:2] for item in avail_raw_data]

        mock_comm = mock.Mock()
        mock_comm.bcast.return_value = [avail_raw_data, avail_stations]

        c = correlate.Correlator()
        c.avail_raw_data = avail_raw_data
        c.station = avail_stations
        c.logger = logging_mock
        c.rank = 0
        c.comm = mock_comm
        c.options = co_options

        self.assertRaises(ValueError, c._check_joint_norm)

    @mock.patch.object(correlate.Correlator, '__init__')
    def test_check_joint_norm_fdsncomps(
            self, mock_init, makedirs_mock, logging_mock, open_mock):
        mock_init.return_value = None
        joint_norm = True
        co_options = deepcopy(self.options["co"])
        co_options['joint_norm'] = joint_norm
        co_options['corr_args']['FDpreProcessing'][0]['args'] = {
            'joint_norm': joint_norm}

        mock_comm = mock.Mock()

        avail_raw_data = sorted(
            [["NET", "STA%s" % i, "", "HH%s" % c]
                for i in "1" for c in "NE4"])
        avail_stations = [item[:2] for item in avail_raw_data]

        mock_comm.bcast.return_value = [avail_raw_data, avail_stations]

        c = correlate.Correlator()
        c.avail_raw_data = avail_raw_data
        c.station = avail_stations
        c.logger = logging_mock
        c.rank = 0
        c.comm = mock_comm
        c.options = co_options

        self.assertRaises(ValueError, c._check_joint_norm)

    @mock.patch.object(correlate.Correlator, '__init__')
    def test_check_joint_norm_pop2cstations(
            self, mock_init, makedirs_mock, logging_mock, open_mock):
        mock_init.return_value = None
        joint_norm = True
        co_options = deepcopy(self.options["co"])
        co_options['joint_norm'] = joint_norm
        co_options['corr_args']['FDpreProcessing'][0]['args'] = {
            'joint_norm': joint_norm}

        for popi in range(9):
            with self.subTest(popi=popi):
                avail_raw_data = sorted(
                    [["NET", "STA%s" % i, "", "HH%s" % c]
                        for i in "123" for c in "NEZ"])
                avail_raw_data.pop(popi)
                avail_stations = np.unique(
                    [item[:2] for item in avail_raw_data], axis=0).tolist()
                icut = popi - popi % 3
                avail_raw_data_new = [avail_raw_data[i] for i
                                      in range(len(avail_raw_data))
                                      if i not in range(icut, icut+2)]
                avail_stations_new = np.unique(
                    [item[:2] for item in avail_raw_data_new], axis=0).tolist()

                c = correlate.Correlator()

                c.avail_raw_data = avail_raw_data
                c.station = avail_stations
                c.logger = logging_mock
                c.comm = MPI.COMM_WORLD
                c.rank = c.comm.Get_rank()
                c.options = co_options
                c._check_joint_norm()

                self.assertListEqual(c.station, avail_stations_new)
                self.assertListEqual(c.avail_raw_data, avail_raw_data_new)

    @mock.patch.object(correlate.Correlator, '__init__')
    def test_get_row_index_per_core_jointnorm_False(
            self, mock_init, makedirs_mock, logging_mock, open_mock):
        mock_init.return_value = None
        co_options = deepcopy(self.options["co"])
        co_options['joint_norm'] = False

        c = correlate.Correlator()
        c.options = co_options
        c.psize = 3

        A = np.empty((10, 10))

        for rank in range(c.psize):
            with self.subTest(rank=rank):
                c.rank = rank
                ind = c._get_row_index_per_core(A)
                ind_true = (np.arange(A.shape[0])*c.psize // A.shape[0]
                            == rank)
                self.assertTrue(np.all(ind == ind_true))

    @mock.patch.object(correlate.Correlator, '__init__')
    def test_get_row_index_per_core_jointnorm_True(
            self, mock_init, makedirs_mock, logging_mock, open_mock):
        mock_init.return_value = None
        co_options = deepcopy(self.options["co"])
        co_options['joint_norm'] = True

        c = correlate.Correlator()
        c.options = co_options
        c.psize = 3

        A = np.empty((10, 10))
        c.rank = 0
        self.assertRaises(
            ValueError, c._get_row_index_per_core, A)

        A = np.empty((12, 10))
        for rank in range(c.psize):
            with self.subTest(rank=rank):
                c.rank = rank
                ind = c._get_row_index_per_core(A)
                ind_true = (np.arange(A.shape[0]/3)*c.psize
                            // (A.shape[0]/3)).repeat(3) == rank
                self.assertTrue(np.all(ind == ind_true))
                self.assertTrue(A[ind].shape[0] % 3 == 0)


class TestStToNpArray(unittest.TestCase):
    def setUp(self):
        self.st = read()

    def test_result_shape(self):
        A, _ = correlate.st_to_np_array(self.st, self.st[0].stats.npts)
        self.assertEqual(A.shape, (self.st.count(), self.st[0].stats.npts))

    def test_deleted_data(self):
        _, st = correlate.st_to_np_array(self.st, self.st[0].stats.npts)
        for tr in st:
            with self.assertRaises(AttributeError):
                type(tr.data)

    def test_result(self):
        A, _ = correlate.st_to_np_array(self.st.copy(), self.st[0].stats.npts)
        for ii, tr in enumerate(self.st):
            self.assertTrue(np.allclose(tr.data, A[ii]))


class TestCompareExistingData(unittest.TestCase):
    def setUp(self):
        st = read()
        st.sort(keys=['channel'])
        tr0 = st[0]
        tr1 = st[1]
        ex_d = {}
        net0 = tr0.stats.network
        net1 = tr1.stats.network
        stat0 = tr0.stats.station
        stat1 = tr1.stats.station
        loc0 = tr0.stats.location
        loc1 = tr1.stats.location
        cha0 = tr0.stats.channel
        cha1 = tr1.stats.channel
        ex_d = {
            f'{net0}.{stat0}': {f'{net1}.{stat1}': {f'{loc0}-{loc1}': {
                f'{cha0}-{cha1}': [tr0.stats.starttime.format_fissures()]}}}}

        self.ex_d = ex_d
        self.tr0 = tr0
        self.tr1 = tr1

    def test_existing(self):
        self.assertTrue(correlate._compare_existing_data(
            self.ex_d, self.tr0, self.tr1))

    def test_not_in_db(self):
        ttr = self.tr0.copy()
        ttr.stats.starttime += 1
        self.assertFalse(correlate._compare_existing_data(
            {}, ttr, self.tr1))

    def test_key_error_handler(self):
        self.assertFalse(correlate._compare_existing_data(
            {}, self.tr0, self.tr1))


class TestCalcCrossCombis(unittest.TestCase):
    def setUp(self):
        channels = ['HHZ', 'HHE', 'HHN']
        net = ['TOTALLY']*3 + ['RANDOM']*3
        stat = ['RANDOM', 'BUT', 'SA', 'ME', 'LEN', 'GTH']
        # randomize the length of each a bit
        randfact = np.random.randint(2, 6)
        xtran = []
        xtras = []
        for ii in range(2, randfact):
            for n, s in zip(net, stat):
                xtran.append(n*ii)
                xtras.append(s*ii)
        net.extend(xtran)
        stat.extend(xtras)

        self.st = Stream()
        for station, network in zip(stat, net):
            for ch in channels:
                stats = AttribDict(
                    network=network, station=station, channel=ch)
                self.st.append(Trace(header=stats))
        self.N_stat = len(stat)
        self.N_chan = len(channels)

    def test_result_betw_stations(self):
        # easiest probably to check the length
        # in this cas \Sum_1^N (N-n)*M^2 where N is the number of stations
        # and M the number of channels
        expected_len = sum([(self.N_stat-n)*self.N_chan**2
                            for n in range(1, self.N_stat)])

        self.assertEqual(expected_len, len(correlate.calc_cross_combis(
            self.st, {}, method='betweenStations')))

    def test_result_betw_components(self):
        # easiest probably to check the length
        # Here, we are looking for the same station but different component
        expected_len = sum([(self.N_chan-n)*self.N_stat
                            for n in range(1, self.N_chan)])

        self.assertEqual(expected_len, len(correlate.calc_cross_combis(
            self.st, {}, method='betweenComponents')))

    def test_result_auto_components(self):
        expected_len = self.st.count()
        self.assertEqual(expected_len, len(correlate.calc_cross_combis(
            self.st, {}, method='autoComponents')))

    def test_result_all_simple(self):
        expected_len = sum([self.st.count()-n
                            for n in range(0, self.st.count())])
        self.assertEqual(expected_len, len(correlate.calc_cross_combis(
            self.st, {}, method='allSimpleCombinations')))

    def test_result_all_combis(self):
        expected_len = sum([self.st.count()-n
                            for n in range(0, self.st.count())])
        with self.assertWarns(DeprecationWarning) as w:
            res = correlate.calc_cross_combis(
                self.st, {}, method='allCombinations')
        self.assertIn("allSimpleCombinations", w.warning.args[0])
        self.assertEqual(expected_len, len(res))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            correlate.calc_cross_combis(self.st, {}, method='blablub')

    def test_empty_stream(self):
        self.assertWarns(UserWarning,
                         correlate.calc_cross_combis, Stream(), {},
                         method='allCombinations')
        # with self.assertWarns(UserWarning) as w:
        #     correlate.calc_cross_combis(Stream(), {},
        #                                 method='allSimpleCombinations')
        # self.assertIn("found no combination", w.warning.args[0])

    @mock.patch('seismic.correlate.correlate._compare_existing_data')
    def test_existing_db(self, compare_mock):
        compare_mock.return_value = True
        for m in [
            'betweenStations', 'betweenComponents', 'autoComponents',
                'allSimpleCombinations']:
            with warnings.catch_warnings(record=True) as w:
                self.assertEqual(0, len(correlate.calc_cross_combis(
                    self.st, {}, method=m)))
                self.assertEqual(len(w), 1)

    def test_rcombis(self):
        xlen = np.random.randint(1, 6)
        rcombis = []
        self.st.sort(keys=['network', 'station', 'channel'])
        for ii, tr in enumerate(self.st):
            if len(rcombis) == xlen:
                break
            for jj in range(ii+1, len(self.st)):
                tr1 = self.st[jj]
                n = tr.stats.network
                n2 = tr1.stats.network
                s = tr.stats.station
                s2 = tr1.stats.station
                if n != n2 or s != s2:
                    rcombis.append('%s-%s.%s-%s' % (n, n2, s, s2))
                    # remove duplicates
                    rcombis = list(set(rcombis))
                if len(rcombis) == xlen:
                    break
        expected_len = xlen*self.N_chan**2
        self.assertEqual(expected_len, len(correlate.calc_cross_combis(
            self.st, {}, method='betweenStations', rcombis=rcombis)))

    def test_rcombis_with_cha(self):
        xlen = np.random.randint(1, 6)
        rcombis = []
        for ii, tr in enumerate(self.st):
            if len(rcombis) == xlen:
                break
            for jj in range(ii+1, len(self.st)):
                tr1 = self.st[jj]
                n = tr.stats.network
                n2 = tr1.stats.network
                s = tr.stats.station
                s2 = tr1.stats.station
                ch1 = tr.stats.channel
                ch2 = tr1.stats.channel
                if n != n2 or s != s2:
                    rcombis.append(
                        '%s-%s.%s-%s.%s-%s' % (n, n2, s, s2, ch1, ch2))
                    # remove duplicates
                    rcombis = list(set(rcombis))
                if len(rcombis) == xlen:
                    break
        expected_len = xlen
        self.assertEqual(expected_len, len(correlate.calc_cross_combis(
            self.st, {}, method='betweenStations', rcombis=rcombis)))

    def test_rcombis_not_available(self):
        rcombis = ['is-not.in-db']
        expected_len = 0
        self.assertEqual(expected_len, len(correlate.calc_cross_combis(
            self.st, {}, method='betweenStations', rcombis=rcombis)))


class TestIsInXcombis(unittest.TestCase):
    def test_in_xcombis(self):
        id1 = 'A.C.loc.E'
        id2 = 'B.D.loc.F'
        rcombis = ['A-B.C-D.E-F', 'G-H.I-J.K-L']
        self.assertTrue(correlate.is_in_xcombis(id1, id2, rcombis))

    def test_in_xcombis_other_way(self):
        id2 = 'A.C.loc.E'
        id1 = 'B.D.loc.F'
        rcombis = ['A-B.C-D.E-F', 'G-H.I-J.K-L']
        self.assertTrue(correlate.is_in_xcombis(id1, id2, rcombis))

    def test_in_xcombis_no_chan(self):
        id2 = 'A.C.loc.E'
        id1 = 'B.D.loc.F'
        rcombis = ['A-B.C-D', 'G-H.I-J']
        self.assertTrue(correlate.is_in_xcombis(id1, id2, rcombis))

    def test_not_in_xcombis(self):
        id1 = 'A.D.E.F'
        id2 = 'G-H..J.K-L'
        rcombis = ['A-B.C-D.E-F', 'M-N.O-P.Q-R']
        self.assertFalse(correlate.is_in_xcombis(id1, id2, rcombis))

    def test_empty_xcombis(self):
        id2 = 'A.C.loc.E'
        id1 = 'B.D.loc.F'
        rcombis = []
        self.assertFalse(correlate.is_in_xcombis(id1, id2, rcombis))


class TestSortCombnameAlphabetically(unittest.TestCase):
    def test_retain_input(self):
        net0 = 'A'
        net1 = 'B'
        stat0 = 'Z'
        stat1 = 'C'
        exp_result = ([net0, net1], [stat0, stat1], ['', ''], ['', ''])
        self.assertEqual(
            correlate.sort_comb_name_alphabetically(net0, stat0, net1, stat1),
            exp_result)

    def test_flip_input(self):
        net0 = 'B'
        net1 = 'A'
        stat0 = 'Z'
        stat1 = 'C'
        exp_result = ([net1, net0], [stat1, stat0], ['', ''], ['', ''])
        self.assertEqual(
            correlate.sort_comb_name_alphabetically(net0, stat0, net1, stat1),
            exp_result)

    def test_between_comps(self):
        net0 = 'A'
        net1 = 'A'
        stat0 = 'Z'
        stat1 = 'Z'
        cha0 = 'B'
        cha1 = 'A'
        exp_result = ([net0, net1], [stat0, stat1], ['', ''], [cha1, cha0])
        self.assertEqual(
            correlate.sort_comb_name_alphabetically(
                net0, stat0, net1, stat1, channel1=cha0, channel2=cha1),
            exp_result)

    def test_between_locs(self):
        net0 = 'A'
        net1 = 'A'
        stat0 = 'Z'
        stat1 = 'Z'
        loc0 = 'B'
        loc1 = 'A'
        exp_result = ([net0, net1], [stat0, stat1], [loc1, loc0], ['', ''])
        self.assertEqual(
            correlate.sort_comb_name_alphabetically(
                net0, stat0, net1, stat1, loc0, loc1),
            exp_result)

    def test_between_comps_no_flip(self):
        net0 = 'A'
        net1 = 'A'
        stat0 = 'Z'
        stat1 = 'Z'
        cha1 = 'B'
        cha0 = 'A'
        exp_result = ([net0, net1], [stat0, stat1], ['', ''], [cha0, cha1])
        self.assertEqual(
            correlate.sort_comb_name_alphabetically(
                net0, stat0, net1, stat1, channel1=cha0, channel2=cha1),
            exp_result)

    def test_wrong_arg_type(self):
        net0 = 1
        net1 = 'A'
        stat0 = 'Z'
        stat1 = 'C'
        with self.assertRaises(TypeError):
            correlate.sort_comb_name_alphabetically(net0, stat0, net1, stat1)


class TestComputeNetworkStationCombinations(unittest.TestCase):
    def setUp(self):
        self.nlist = ['A', 'A']
        self.slist = ['B', 'C']

    def test_between_stations_0(self):
        exp_result = (['A-A'], ['B-C'])
        self.assertEqual(
            correlate.compute_network_station_combinations(
                self.nlist, self.slist),
            exp_result)

    def test_between_stations_1(self):
        exp_result = ([], [])
        self.assertEqual(
            correlate.compute_network_station_combinations(
                self.nlist, self.nlist),
            exp_result)

    def test_between_stations_2(self):
        exp_result = (['B-C'], ['A-A'])
        self.assertEqual(
            correlate.compute_network_station_combinations(
                self.slist, self.nlist),
            exp_result)

    def test_between_components(self):
        exp_result = (['A-A', 'A-A'], ['B-B', 'C-C'])
        for m in ['betweenComponents', 'autoComponents']:
            self.assertEqual(
                correlate.compute_network_station_combinations(
                    self.nlist, self.slist, method=m),
                exp_result)

    def test_all_simple_combis(self):
        exp_result = (['A-A', 'A-A', 'A-A'], ['B-B', 'B-C', 'C-C'])
        self.assertEqual(
            correlate.compute_network_station_combinations(
                self.nlist, self.slist, method='allSimpleCombinations'),
            exp_result)

    def test_all_combis(self):
        exp_result = (['A-A', 'A-A', 'A-A'], ['B-B', 'B-C', 'C-C'])
        with warnings.catch_warnings(record=True):
            self.assertEqual(
                correlate.compute_network_station_combinations(
                    self.nlist, self.slist, method='allCombinations'),
                exp_result)
            relevant_warning = warnings.simplefilter("error", ValueError)
            self.assertIsNone(relevant_warning)
            # self.assertEqual(len(w), 0)

    def test_rcombis(self):
        exp_result = (['A-A'], ['B-C'])
        nlist = ['A', 'A', 'B']
        slist = ['B', 'C', 'D']
        rcombis = ['A-A.B-C']
        self.assertEqual(
            correlate.compute_network_station_combinations(
                nlist, slist, combis=rcombis), exp_result)

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            correlate.compute_network_station_combinations(
                self.nlist, self.slist, method='b')


class TestPreProcessStream(unittest.TestCase):
    def setUp(self):
        self.st = read()
        for tr in self.st:
            del tr.stats.response
        self.kwargs = {
            'store_client': mock.MagicMock(),
            'inv': read_inventory(),
            'startt': self.st[0].stats.starttime,
            'endt': self.st[0].stats.endtime,
            'taper_len': 0,
            'sampling_rate': 25,
            'remove_response': False,
            'subdivision': {'corr_len': 20}}
        self.kwargs['store_client'].inventory = read_inventory()

    def test_empty_stream(self):
        self.assertEqual(
            correlate.preprocess_stream(Stream(), **self.kwargs), Stream())

    def test_pad(self):
        kwargs = deepcopy(self.kwargs)
        kwargs['startt'] -= 10
        kwargs['endt'] += 10
        st = correlate.preprocess_stream(self.st.copy(), **kwargs)
        self.assertAlmostEqual(
            kwargs['startt'], st[0].stats.starttime, delta=st[0].stats.delta/2)
        self.assertAlmostEqual(
            kwargs['endt'], st[0].stats.endtime, delta=st[0].stats.delta/2)

    def test_trim(self):
        kwargs = deepcopy(self.kwargs)
        kwargs['startt'] += 5
        kwargs['endt'] -= 5
        st = correlate.preprocess_stream(self.st.copy(), **kwargs)
        self.assertAlmostEqual(
            kwargs['startt'], st[0].stats.starttime, delta=st[0].stats.delta/2)
        self.assertAlmostEqual(
            kwargs['endt'], st[0].stats.endtime, delta=st[0].stats.delta/2)

    def test_discard_short(self):
        kwargs = deepcopy(self.kwargs)
        kwargs['startt'] += 15
        kwargs['endt'] -= 5
        kwargs['subdivision']['corr_len'] = 200
        st = correlate.preprocess_stream(self.st.copy(), **kwargs)
        self.assertFalse(st.count())

    def test_remove_resp(self):
        kwargs = deepcopy(self.kwargs)
        kwargs['remove_response'] = True
        st = correlate.preprocess_stream(self.st.copy(), **kwargs)
        self.assertTrue(np.any(
            'remove_response' in processing_step
            for processing_step in st[0].stats.processing))

    def test_taper(self):
        kwargs = deepcopy(self.kwargs)
        kwargs['remove_response'] = True
        st = correlate.preprocess_stream(self.st.copy(), **kwargs)

        kwargs = deepcopy(kwargs)
        kwargs['remove_response'] = True
        kwargs['taper_len'] = 5
        stt = correlate.preprocess_stream(self.st.copy(), **kwargs)
        self.assertFalse(np.allclose(stt[0].data, st[0].data))

    def test_no_inv(self):
        kwargs = deepcopy(self.kwargs)
        kwargs['remove_response'] = True
        kwargs['inv'] = None
        sc_mock = mock.MagicMock()
        sc_mock.rclient.get_stations.return_value = self.kwargs['inv']
        kwargs['store_client'] = sc_mock
        st = correlate.preprocess_stream(self.st.copy(), **kwargs)
        self.assertTrue(np.any(
            'remove_response' in processing_step
            for processing_step in st[0].stats.processing))
        sc_mock.rclient.get_stations.assert_called_once()

    @mock.patch('seismic.correlate.correlate.func_from_str')
    def test_additional_preprofunc(self, ffs_mock):
        return_func = mock.MagicMock()
        return_func.return_value = self.st.copy()
        ffs_mock.return_value = return_func
        kwargs = deepcopy(self.kwargs)
        kwargs['preProcessing'] = [{'function': 'bla', 'args': {'arg': 'bla'}}]
        correlate.preprocess_stream(self.st, **kwargs)
        ffs_mock.assert_called_once_with('bla')
        return_func.assert_called_once_with(
            mock.ANY, arg='bla')


class TestGenCorrInc(unittest.TestCase):
    def setUp(self):
        st = read()
        self.st = st
        self.rl = st[0].stats.npts/st[0].stats.sampling_rate

    def test_empty(self):
        subdivision = {'corr_inc': self.rl/10, 'corr_len': 10}
        x = list(correlate.generate_corr_inc(Stream(), subdivision, self.rl))
        self.assertEqual(len(x), 10)
        for st in x:
            self.assertEqual(Stream(), st)

    def test_result(self):
        subdivision = {'corr_inc': self.rl/10, 'corr_len': 10}
        x = list(correlate.generate_corr_inc(self.st, subdivision, self.rl))
        self.assertEqual(len(x), 10)
        for st in x:
            for tr in st:
                self.assertAlmostEqual(
                    10, tr.stats.npts/tr.stats.sampling_rate)

    def test_pad(self):
        subdivision = {'corr_inc': self.rl/10, 'corr_len': 45}
        x = list(correlate.generate_corr_inc(self.st, subdivision, self.rl))
        self.assertEqual(len(x), 10)
        for st in x:
            for tr in st:
                self.assertAlmostEqual(
                    45, tr.stats.npts/tr.stats.sampling_rate)


class TestCorrelatorFilterByRcombis(unittest.TestCase):
    def setUp(self):
        self.corr = correlate.Correlator(None, None)


class TestCheckForMissingChannels(unittest.TestCase):
    def test_streams(self):
        avail_channels = sorted(
            [["NET", "STA%s" % i, "", "HH%s" % c]
                for i in "123" for c in "ZNE"])
        st = Stream()
        for n, s, l, c in avail_channels[:4]:
            stats = dict(network=n, station=s,
                         location=l, channel=c,
                         sampling_rate=10,)
            tr = Trace(np.random.rand(1000), stats)
            st.append(tr)

        st_out = st.copy()
        for n, s, l, c in avail_channels[4:6]:
            stats = dict(network=n, station=s,
                         location=l, channel=c,
                         sampling_rate=10,)
            tr = Trace(np.zeros(1000), stats)
            st_out.append(tr)
        correlate.check_for_missing_channels(st, avail_channels)
        self.assertEqual(st, st_out)

        # Check if trace order corresponds to avail_channels
        nslc = [tr.id.split(".") for tr in st]
        self.assertEqual(nslc, avail_channels[:6])


if __name__ == "__main__":
    unittest.main()
