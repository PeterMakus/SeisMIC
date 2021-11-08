'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 27th May 2021 04:27:14 pm
Last Modified: Monday, 8th November 2021 05:25:27 pm
'''
from copy import deepcopy
import unittest
import warnings
from unittest import mock
import os

import numpy as np
from obspy import read, Stream, Trace
from obspy.core import AttribDict
from obspy.core.inventory.inventory import read_inventory
import yaml

from seismic.correlate import correlate
from seismic.trace_data.waveform import Store_Client


class TestCorrrelator(unittest.TestCase):
    def setUp(self):
        os.path.join
        # using the example parameters has the nice side effect that
        # the parameter file is tested as well
        self.param_example = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'params_example.yaml')
        with open(self.param_example) as file:
            self.options = yaml.load(file, Loader=yaml.FullLoader)

    @mock.patch('seismic.correlate.correlate.yaml.load')
    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_init_options_from_yaml(
            self, makedirs_mock, logging_mock, open_mock, yaml_mock):
        yaml_mock.return_value = self.options
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        c = correlate.Correlator(sc_mock, self.param_example)
        self.assertDictEqual(self.options['co'], c.options)
        mkdir_calls = [
            mock.call(os.path.join(
                self.options['proj_dir'], self.options['co']['subdir']),
                exist_ok=True),
            mock.call(os.path.join(
                self.options['proj_dir'], self.options['log_subdir']),
                exist_ok=True)]
        makedirs_mock.assert_has_calls(mkdir_calls)
        open_mock.assert_any_call(self.param_example)

    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_stat_net_str(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = ['bla']
        options['net']['station'] = ['blub']
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        c = correlate.Correlator(sc_mock, options)
        self.assertDictEqual(self.options['co'], c.options)
        self.assertListEqual(c.station, [['bla', 'blub']])

    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_net_wc_stat_str(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = '*'
        options['net']['station'] = ['blub']
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        with self.assertRaises(ValueError):
            correlate.Correlator(sc_mock, options)

    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_net_wc(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = '*'
        options['net']['station'] = '*'
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [['lala', 'lolo']]
        c = correlate.Correlator(sc_mock, options)
        self.assertListEqual(c.station, [['lala', 'lolo']])

    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_stat_wc(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = 'lala'
        options['net']['station'] = '*'
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [['lala', 'lolo']]
        c = correlate.Correlator(sc_mock, options)
        self.assertListEqual(c.station, [['lala', 'lolo']])
        sc_mock.get_available_stations.assert_called_once_with('lala')

    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_stat_wc_net_list(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = ['lala', 'lolo']
        options['net']['station'] = '*'
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [['lala', 'lolo']]
        c = correlate.Correlator(sc_mock, options)
        self.assertListEqual(
            c.station, [['lala', 'lolo'], ['lala', 'lolo']])
        calls = [mock.call('lala'), mock.call('lolo')]
        sc_mock.get_available_stations.assert_has_calls(calls)

    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_list_len_diff(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = [1, 2, 3]
        options['net']['station'] = ['blub', 'blib']
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        with self.assertRaises(ValueError):
            correlate.Correlator(sc_mock, options)

    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_stat_net_list(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = ['lala', 'lolo']
        options['net']['station'] = ['le', 'li']
        sc_mock = mock.Mock(Store_Client)
        c = correlate.Correlator(sc_mock, options)
        self.assertListEqual(
            c.station, [['lala', 'le'], ['lolo', 'li']])

    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_other_wrong(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = 1
        options['net']['station'] = ['blub']
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = []
        with self.assertRaises(ValueError):
            correlate.Correlator(sc_mock, options)

    @mock.patch('seismic.correlate.correlate.mu.filter_stat_dist')
    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_find_interstat_dis(
            self, makedirs_mock, logging_mock, open_mock, statd_mock):
        options = deepcopy(self.options)
        options['net']['network'] = '*'
        options['net']['station'] = '*'
        options['co']['combination_method'] = 'betweenStations'
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [
            ['lala', 'lolo'], ['lala', 'lili']]
        c = correlate.Correlator(sc_mock, options)

        sc_mock.select_inventory_or_load_remote.side_effect = [
            'a', 'b', 'c', 'd', 'e', 'f']
        statd_mock.return_value = True
        c.find_interstat_dist(10000)
        sc_mock.read_inventory.assert_called_once()
        sc_mock.select_inventory_or_load_remote.assert_called_with(
            'lala', 'lili')
        self.assertListEqual(
            c.rcombis,
            [
                'lala-lala.lolo-lolo',
                'lala-lala.lolo-lili', 'lala-lala.lili-lili'])

    @mock.patch('builtins.open')
    @mock.patch('seismic.correlate.correlate.logging')
    @mock.patch('seismic.correlate.correlate.os.makedirs')
    def test_find_interstat_dis_wrong_method(
            self, makedirs_mock, logging_mock, open_mock):
        options = deepcopy(self.options)
        options['net']['network'] = '*'
        options['net']['station'] = '*'
        options['co']['combination_method'] = 'betweenComponents'
        sc_mock = mock.Mock(Store_Client)
        sc_mock.get_available_stations.return_value = [
            ['lala', 'lolo'], ['lala', 'lili']]
        c = correlate.Correlator(sc_mock, options)
        with self.assertRaises(ValueError):
            c.find_interstat_dist(100)


class TestStToNpArray(unittest.TestCase):
    def setUp(self):
        self.st = read()

    def test_result_shape(self):
        A, _ = correlate.st_to_np_array(self.st, self.st[0].stats.npts)
        self.assertEqual(A.shape, (self.st[0].stats.npts, self.st.count()))

    def test_deleted_data(self):
        _, st = correlate.st_to_np_array(self.st, self.st[0].stats.npts)
        for tr in st:
            with self.assertRaises(AttributeError):
                print(tr.data)

    def test_result(self):
        A, _ = correlate.st_to_np_array(self.st.copy(), self.st[0].stats.npts)
        for ii, tr in enumerate(self.st):
            self.assertTrue(np.allclose(tr.data, A[:, ii]))


class TestCompareExistingData(unittest.TestCase):
    def setUp(self):
        st = read()
        st.sort(keys=['channel'])
        tr0 = st[0]
        tr1 = st[1]
        ex_d = {}
        ex_d['%s.%s' % (tr0.stats.network, tr0.stats.station)] = {}
        ex_d[
            '%s.%s' % (tr0.stats.network, tr0.stats.station)][
                '%s.%s' % (tr1.stats.network, tr1.stats.station)] = {}
        ex_d[
            '%s.%s' % (tr0.stats.network, tr0.stats.station)][
                '%s.%s' % (tr1.stats.network, tr1.stats.station)] = {}
        ex_d[
            '%s.%s' % (tr0.stats.network, tr0.stats.station)][
                '%s.%s' % (tr1.stats.network, tr1.stats.station)][
            '%s-%s' % (
                tr0.stats.channel, tr1.stats.channel)] = [
                    tr0.stats.starttime]
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
        expected_len = self.st.count()**2
        self.assertEqual(expected_len, len(correlate.calc_cross_combis(
            self.st, {}, method='allCombinations')))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            correlate.calc_cross_combis(self.st, {}, method='blablub')

    def test_empty_stream(self):
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual([], correlate.calc_cross_combis(
                Stream(), {}, method='allCombinations'))
            self.assertEqual(len(w), 1)

    @mock.patch('seismic.correlate.correlate._compare_existing_data')
    def test_existing_db(self, compare_mock):
        compare_mock.return_value = True
        for m in [
            'betweenStations', 'betweenComponents', 'autoComponents',
                'allSimpleCombinations', 'allCombinations']:
            with warnings.catch_warnings(record=True) as w:
                self.assertEqual(0, len(correlate.calc_cross_combis(
                    self.st, {}, method=m)))
                self.assertEqual(len(w), 1)

    def test_rcombis(self):
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
                if n != n2 or s != s2:
                    rcombis.append('%s-%s.%s-%s' % (n, n2, s, s2))
                    # remove duplicates
                    rcombis = list(set(rcombis))
                if len(rcombis) == xlen:
                    break
        expected_len = xlen*self.N_chan**2
        self.assertEqual(expected_len, len(correlate.calc_cross_combis(
            self.st, {}, method='betweenStations', rcombis=rcombis)))

    def test_rcombis_not_available(self):
        rcombis = ['is-not.in-db']
        expected_len = 0
        self.assertEqual(expected_len, len(correlate.calc_cross_combis(
            self.st, {}, method='betweenStations', rcombis=rcombis)))


class TestSortCombnameAlphabetically(unittest.TestCase):
    def test_retain_input(self):
        net0 = 'A'
        net1 = 'B'
        stat0 = 'Z'
        stat1 = 'C'
        exp_result = ([net0, net1], [stat0, stat1])
        self.assertEqual(
            correlate.sort_comb_name_alphabetically(net0, stat0, net1, stat1),
            exp_result)

    def test_flip_input(self):
        net0 = 'B'
        net1 = 'A'
        stat0 = 'Z'
        stat1 = 'C'
        exp_result = ([net1, net0], [stat1, stat0])
        self.assertEqual(
            correlate.sort_comb_name_alphabetically(net0, stat0, net1, stat1),
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
        exp_result = (
            ['A-A', 'A-A', 'A-A', 'A-A'], ['B-B', 'B-C', 'B-C', 'C-C'])
        self.assertEqual(
            correlate.compute_network_station_combinations(
                self.nlist, self.slist, method='allCombinations'),
            exp_result)

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
            'store_client': None,
            'inv': read_inventory(),
            'startt': self.st[0].stats.starttime,
            'endt': self.st[0].stats.endtime,
            'taper_len': 0,
            'sampling_rate': 25,
            'remove_response': False,
            'subdivision': {'corr_len': 20}}

    def test_empty_stream(self):
        self.assertEqual(
            correlate.preprocess_stream(Stream(), **self.kwargs), Stream())

    def test_wrong_sr(self):
        x = np.random.randint(1, 100)
        sr = self.st[0].stats.sampling_rate + x
        kwargs = deepcopy(self.kwargs)
        kwargs['sampling_rate'] = sr
        with self.assertRaises(ValueError):
            correlate.preprocess_stream(self.st, **kwargs)

    def test_decimate(self):
        st = correlate.preprocess_stream(self.st.copy(), **self.kwargs)
        self.assertEqual(25, st[0].stats.sampling_rate)

    def test_resample(self):
        kwargs = deepcopy(self.kwargs)
        kwargs['sampling_rate'] = 23
        st = correlate.preprocess_stream(self.st.copy(), **kwargs)
        self.assertEqual(23, st[0].stats.sampling_rate)

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
        kwargs['endt'] -= 15
        kwargs['subdivision']['corr_len'] = 200
        st = correlate.preprocess_stream(self.st.copy(), **kwargs)
        self.assertFalse(st.count())

    def test_remove_resp(self):
        kwargs = deepcopy(self.kwargs)
        kwargs['remove_response'] = True
        st = correlate.preprocess_stream(self.st.copy(), **kwargs)
        self.assertIn('remove_response', st[0].stats.processing[-2])
        # Check whether response was attached
        self.assertTrue(st[0].stats.response)

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
        self.assertIn('remove_response', st[0].stats.processing[-2])
        # Check whether response was attached
        self.assertTrue(st[0].stats.response)
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


if __name__ == "__main__":
    unittest.main()
