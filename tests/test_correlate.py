'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 27th May 2021 04:27:14 pm
Last Modified: Tuesday, 20th July 2021 03:57:49 pm
'''
import unittest
import warnings
from unittest import mock

import numpy as np
from obspy import read, Stream, Trace
from obspy.core import AttribDict
from scipy.fftpack import next_fast_len
from scipy.signal.windows import gaussian

from miic3.correlate import correlate


# class TestCorrrelator(unittest.TestCase):
# have not figured how to test this


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
        # Also, randomize the len of the channels
        # The channels are combined if the last letter is different
        randfact = np.random.randint(-2, 3)
        xtrac = ['R', 'L', 'P']
        if randfact <= 0:
            channels = channels[:randfact]
        else:
            channels.extend(xtrac[:randfact])
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

    @mock.patch('miic3.correlate.correlate._compare_existing_data')
    def test_existing_db(self, compare_mock):
        compare_mock.return_value = True
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(0, len(correlate.calc_cross_combis(
                self.st, {}, method='betweenStations')))
            self.assertEqual(len(w), 1)


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


class TestComputeNetworkStationCombinations(unittest.TestCase):
    def setUp(self):
        self.nlist = ['A', 'A']
        self.slist = ['B', 'C']

    def test_between_stations_0(self):
        exp_result = ([['A', 'A']], [['B', 'C']])
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
        exp_result = ([['B', 'C']], [['A', 'A']])
        self.assertEqual(
            correlate.compute_network_station_combinations(
                self.slist, self.nlist),
            exp_result)


if __name__ == "__main__":
    unittest.main()
