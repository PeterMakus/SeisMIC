'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 27th May 2021 04:27:14 pm
Last Modified: Friday, 11th June 2021 02:16:57 pm
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


class TestZeroPadding(unittest.TestCase):
    def setUp(self):
        self.params = {'sampling_rate': 25, 'lengthToSave': 200}
        self.A = np.empty(
            (np.random.randint(100, 666), np.random.randint(2, 45)))

    def test_result_next_fast_len(self):
        expected_len = next_fast_len(self.A.shape[0])
        self.assertEqual(correlate.zeroPadding(
            self.A, {'type': 'nextFastLen'}, self.params).shape[0],
            expected_len)

    def test_result_avoid_wrap_around(self):
        expected_len = self.A.shape[0] + \
            self.params['sampling_rate'] * self.params['lengthToSave']
        self.assertEqual(correlate.zeroPadding(
            self.A, {'type': 'avoidWrapAround'}, self.params).shape[0],
            expected_len)

    def test_result_avoid_wrap_fast_len(self):
        expected_len = next_fast_len(int(
            self.A.shape[0] +
            self.params['sampling_rate'] * self.params['lengthToSave']))
        self.assertEqual(correlate.zeroPadding(
            self.A, {'type': 'avoidWrapFastLen'}, self.params).shape[0],
            expected_len)

    def test_result_next_fast_len_axis1(self):
        expected_len = next_fast_len(self.A.shape[1])
        self.assertEqual(correlate.zeroPadding(
            self.A, {'type': 'nextFastLen'}, self.params, axis=1).shape[1],
            expected_len)

    def test_result_avoid_wrap_around_axis1(self):
        expected_len = self.A.shape[1] + \
            self.params['sampling_rate'] * self.params['lengthToSave']
        self.assertEqual(correlate.zeroPadding(
            self.A, {'type': 'avoidWrapAround'}, self.params, axis=1).shape[1],
            expected_len)

    def test_result_avoid_wrap_fast_len_axis1(self):
        expected_len = next_fast_len(int(
            self.A.shape[1] +
            self.params['sampling_rate'] * self.params['lengthToSave']))
        self.assertEqual(correlate.zeroPadding(
            self.A,
            {'type': 'avoidWrapFastLen'}, self.params, axis=1).shape[1],
            expected_len)

    def test_weird_axis(self):
        with self.assertRaises(NotImplementedError):
            correlate.zeroPadding(self.A, {}, {}, axis=7)

    def test_higher_dim(self):
        with self.assertRaises(NotImplementedError):
            correlate.zeroPadding(np.ones((3, 3, 3)), {}, {})

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            correlate.zeroPadding(self.A, {'type': 'blub'}, self.params)

    def test_empty_array(self):
        B = np.array([])
        with self.assertRaises(ValueError):
            correlate.zeroPadding(B, {'type': 'nextFastLen'}, self.params)


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


class TestSpectralWhitening(unittest.TestCase):
    def setUp(self):
        dim = (np.random.randint(200, 766), np.random.randint(2, 44))
        self.A = np.random.random(dim) + np.random.random(dim) * 1j

    def test_result(self):
        # Again so straightforward that I wonder whether it makes sense
        # to test this
        expected = self.A/abs(self.A)
        expected[0, :] = 0.j
        self.assertTrue(np.allclose(
            expected, correlate.spectralWhitening(self.A, {}, {})))

    def test_joint_norm_not_possible(self):
        with self.assertRaises(AssertionError):
            correlate.spectralWhitening(
                np.ones((5, 5)), {'joint_norm': True}, {})

    def test_empty_array(self):
        A = np.array([])
        with self.assertRaises(IndexError):
            correlate.spectralWhitening(
                A, {}, {})


class TestSignBitNormalisation(unittest.TestCase):
    # Not much to test here
    def test_result(self):
        np.random.seed(2)
        dim = (np.random.randint(200, 766), np.random.randint(2, 44))
        A = np.random.random(dim)-.5
        Aft = np.fft.rfft(A, axis=0)
        expected_result = np.fft.rfft(np.sign(A), axis=0)
        self.assertTrue(np.allclose(
            expected_result, correlate.FDsignBitNormalization(Aft, {}, {})))


class TestMute(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.params['sampling_rate'] = 25

    def test_return_zeros(self):
        # function is supposed to return zeros if input shorter than
        # the taper length
        npts = np.random.randint(100, 599)
        A = np.ones((npts, np.random.randint(2, 55)))
        args = {}
        args['taper_len'] = self.params['sampling_rate']*(
            npts + np.random.randint(1, 99))
        self.assertTrue(
            np.all(correlate.mute(A, args, self.params) == np.zeros_like(A)))

    def test_taper_len_error(self):
        args = {}
        args['taper_len'] = 0
        A = np.ones((5, 2))
        with self.assertRaises(ValueError):
            correlate.mute(A, args, self.params)

    def test_mute_std(self):
        # testing the actual muting of the bit
        args = {}
        args['taper_len'] = 1
        args['extend_gaps'] = True
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1)).T
        res = correlate.mute(A.copy(), args, self.params)
        self.assertLessEqual(
            res[:, 0].max(axis=0), np.std(A))

    def test_mute_std_factor(self):
        # testing the actual muting of the bit
        args = {}
        args['taper_len'] = 1
        args['extend_gaps'] = True
        args['std_factor'] = np.random.randint(1, 5)
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1)).T
        res = correlate.mute(A.copy(), args, self.params)
        self.assertLessEqual(
            res[:, 0].max(axis=0),
            args['std_factor']*np.std(A))

    def test_mute_absolute(self):
        args = {}
        args['taper_len'] = 1
        args['extend_gaps'] = True
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1)).T
        args['threshold'] = A[:, 0].max(axis=0)/np.random.randint(2, 4)
        res = correlate.mute(A.copy(), args, self.params)
        self.assertLessEqual(
            res[:, 0].max(axis=0), args['threshold'])


class TestNormalizeStd(unittest.TestCase):
    def test_result(self):
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1)).T
        res = correlate.normalizeStandardDeviation(A, {}, {})
        self.assertAlmostEqual(np.std(res, axis=0)[0], 1)

    def test_std_0(self):
        # Feed in DC signal to check this
        A = np.ones((250, 2))
        res = correlate.normalizeStandardDeviation(A.copy(), {}, {})
        self.assertTrue(np.all(res == A))


class TestTDNormalisation(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.params['sampling_rate'] = 25

    def test_win_length_error(self):
        args = {}
        args['windowLength'] = 0
        with self.assertRaises(ValueError):
            correlate.TDnormalization(np.ones((5, 2)), args, self.params)

    # def test_result(self):
    # Gotta think a little about that one
    #     args = {}
    #     args['windowLength'] = 4
    #     args['filter'] = False
    #     A = np.ones(
    #         (np.random.randint(600, 920),
    #             np.random.randint(2, 8)))*np.random.randint(2, 8)
    #     res = correlate.TDnormalization(A.copy(), args, self.params)
    #     self.assertLessEqual(res.max(), 1)


class TestClip(unittest.TestCase):
    def test_result(self):
        args = {}
        args['std_factor'] = np.random.randint(2, 4)
        npts = np.random.randint(400, 749)
        A = np.tile(gaussian(npts, 180), (2, 1)).T
        res = correlate.clip(A.copy(), args, {})
        self.assertAlmostEqual(
            np.std(A, axis=0)[0]*args['std_factor'], abs(res).max(axis=0)[0])

    def test_std_0(self):
        args = {}
        args['std_factor'] = np.random.randint(2, 4)
        A = np.ones((100, 5))
        res = correlate.clip(A.copy(), args, {})
        self.assertTrue(np.all(res == np.zeros_like(A)))


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
