'''
Module to test the asdf handler.
Author: Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 18th March 2021 04:26:31 pm
Last Modified: Friday, 2nd July 2021 11:49:21 am
'''
import unittest
from unittest import mock

import numpy as np
from obspy import read, read_inventory
from obspy.core import AttribDict
from obspy.core.stream import Stream

from miic3.db import asdf_handler


class TestNoiseDB(unittest.TestCase):
    @mock.patch('miic3.db.asdf_handler.ASDFDataSet')
    def setUp(self, asdf_mock) -> None:
        # Example data
        self.st = read()
        self.station = self.st[0].stats.station
        self.network = self.st[0].stats.network
        self.param = {
            "sampling_rate": 20,
            "outfolder": 'location',
            "remove_response": True}
        asdf_mock.return_value.__enter__.return_value = AttribDict(
            waveforms=AttribDict(
                {'%s.%s' % (self.network, self.station): AttribDict({
                    'processed': self.st})}),
            auxiliary_data=AttribDict(
                PreprocessingParameters=AttribDict(
                    param=AttribDict(
                        parameters=self.param))))
        self.ndb = asdf_handler.NoiseDB('outdir', self.network, self.station)

    def test_param(self):
        self.assertEqual(self.param, self.ndb.param)

    @mock.patch('miic3.db.asdf_handler.ASDFDataSet')
    def test_get_active_times(self, asdf_mock):
        """
        Test the result of get_active_times
        """
        asdf_starttime = self.st[0].stats.starttime.timestamp*1e9
        wf_mock = mock.Mock()
        wf_mock.get_waveform_attributes.return_value = \
            {'%s.%s__processed' % (self.network, self.station): {
                'sampling_rate': self.param['sampling_rate'],
                'starttime': asdf_starttime}}
        asdf_mock.return_value.__enter__.return_value = AttribDict(
            waveforms={'%s.%s' % (self.network, self.station): wf_mock})

        act_times = self.ndb.get_active_times()
        self.assertAlmostEqual(act_times[0], self.st[0].stats.starttime)
        self.assertAlmostEqual(
            act_times[1], self.st[0].stats.starttime+24*3600)

    def test_no_file(self):
        """
        Test the result of get_active_times
        """
        with self.assertRaises(FileNotFoundError):
            _ = self.ndb.get_active_times()

    @mock.patch('miic3.db.asdf_handler.ASDFDataSet')
    def test_get_time_window(self, asdf_mock):
        wf_mock = mock.Mock()
        start = self.st[0].stats.starttime - 100
        end = self.st[0].stats.endtime - 5
        start = self.st[0].stats.starttime + np.random.randint(1, 10)
        end = self.st[0].stats.endtime - np.random.randint(1, 10)
        wf_mock.get_waveforms.return_value = self.st.slice(start, end)
        asdf_mock.return_value.__enter__.return_value = wf_mock
        st_out = self.ndb.get_time_window(start, end)
        # Check times
        self.assertAlmostEqual(st_out[0].stats.starttime, start)
        self.assertAlmostEqual(st_out[0].stats.endtime, end)
        wf_mock.get_waveforms.assert_called_once_with(
            self.network, self.station, '*', '*', start, end, 'processed',
            automerge=True)

    @mock.patch('miic3.db.asdf_handler.ASDFDataSet')
    def test_get_partial_data(self, asdf_mock):
        wf_mock = mock.Mock()
        start = self.st[0].stats.starttime - 100
        end = self.st[0].stats.endtime - 5
        wf_mock.get_waveforms.return_value = self.st.slice(start, end)
        asdf_mock.return_value.__enter__.return_value = wf_mock
        st_out = self.ndb.get_time_window(start, end)
        # Check times
        self.assertAlmostEqual(
            st_out[0].stats.starttime, self.st[0].stats.starttime)
        self.assertAlmostEqual(st_out[0].stats.endtime, end)
        wf_mock.get_waveforms.assert_called_once_with(
            self.network, self.station, '*', '*', start, end, 'processed',
            automerge=True)

    @mock.patch('miic3.db.asdf_handler.ASDFDataSet')
    def test_no_data(self, asdf_mock):
        wf_mock = mock.Mock()
        wf_mock.get_waveforms.return_value = Stream()
        asdf_mock.return_value.__enter__.return_value = wf_mock
        start = self.st[0].stats.starttime - 100
        end = self.st[0].stats.starttime - 5
        with self.assertRaises(asdf_handler.NoDataError):
            _ = self.ndb.get_time_window(start, end)

    @mock.patch('miic3.db.asdf_handler.ASDFDataSet')
    def test_get_inventory(self, asdf_mock):
        asdf_mock.return_value.__enter__.return_value = AttribDict(
            waveforms=AttribDict(
                {'%s.%s' % (self.network, self.station): AttribDict({
                    'processed': self.st, 'StationXML': read_inventory()})}),
            auxiliary_data=AttribDict(
                PreprocessingParameters=AttribDict(
                    param=AttribDict(
                        parameters=self.param))))
        self.assertEqual(read_inventory(), self.ndb.get_inventory())


class TestGetAvailableStation(unittest.TestCase):
    def test_no_data(self):
        """
        Tests what happens if there is no data available or the folder does
        not exist.
        """
        with self.assertRaises(FileNotFoundError):
            asdf_handler.get_available_stations(
                '/does/definitely/not/exist', 'For', 'Sure')

    @mock.patch('miic3.db.asdf_handler.glob')
    def test_result(self, glob_mock):
        dir = '/this/path/does/exist/'
        net = ['TOTAL']*3 + ['RANDOM']*3
        stat = ['RANDOM', 'BUT', 'SA', 'ME', 'LEN', 'GTH']
        glob_mock.return_value = [
            dir + a + '.' + b for a, b in zip(net, stat)]
        self.assertEqual(
            asdf_handler.get_available_stations(dir, '*', '*'),
            (net, stat))


if __name__ == "__main__":
    unittest.main()
