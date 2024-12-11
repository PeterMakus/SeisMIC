'''
UnitTests for the waveform module.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 15th March 2021 03:33:25 pm
Last Modified: Monday, 25th November 2024 03:15:26 pm (J. Lehr)
'''

import unittest
from unittest import mock
import os
import warnings
import yaml
from copy import deepcopy

import numpy as np
from obspy import UTCDateTime, Inventory, read_inventory
from seismic.trace_data import waveform
from seismic import trace_data


paramfile = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'params_example.yaml')
with open(paramfile, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


paramfile = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'params_example.yaml')
with open(paramfile, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class TestStoreClient(unittest.TestCase):
    @mock.patch('seismic.trace_data.waveform.os.listdir')
    @mock.patch('seismic.trace_data.waveform.os.path.isdir')
    def setUp(self, isdir_mock, listdir_mock):
        isdir_mock.return_value = True
        listdir_mock.return_value = False
        dir = os.path.join('%smy' % os.path.sep, 'random', 'dir')
        self.outdir = os.path.abspath(os.path.join(dir, waveform.DEFAULT_SDS))
        self.net = 'mynet'
        self.stat = 'mystat'
        self.sc = waveform.Store_Client('testclient', dir, read_only=True)

    def test_not_in_db(self):
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(
                self.sc._get_times(self.net, self.stat), (None, None))
            self.assertEqual(len(w), 1)

    @mock.patch('seismic.trace_data.waveform.glob.glob')
    def test_get_times(self, glob_mock):
        start = UTCDateTime(np.random.randint(0, 3e8))
        end = UTCDateTime(np.random.randint(4e8, 8e8))
        years = np.arange(start.year, end.year+1)
        # days in start year
        startdays = np.arange(start.julday, 366)
        # Days in end year
        enddays = np.arange(1, end.julday+1)
        yeardirs = [os.path.join(
            self.outdir, str(year), self.net, self.stat) for year in years]
        startdaydirs = [os.path.join(
            yeardirs[0], 'HHE.D', "%s.%s.%s.%s.%s.%s.%s" %
            (self.net, self.stat, '00', 'HHE', 'D', years[0], day))
            for day in startdays]

        enddaydirs = [os.path.join(
            yeardirs[-1], 'HHE.D',
            "%s.%s.%s.%s.%s.%s.%s" %
            (self.net, self.stat, '00', 'HHE', 'D', years[-1], day))
            for day in enddays]
        glob_mock.side_effect = [yeardirs, startdaydirs, enddaydirs]
        #
        # Check if the directory is correct
        self.assertEqual(self.sc.sds_root, self.outdir)
        timetup = self.sc._get_times(self.net, self.stat)
        #
        t_control = (
            UTCDateTime(year=start.year, julday=start.julday),
            UTCDateTime(year=end.year, julday=end.julday)+24*3600)
        self.assertEqual(timetup, t_control)

    @mock.patch('seismic.trace_data.waveform.glob.glob')
    def test_get_times_year_change(self, glob_mock):
        # Just checking as there is no julday 366 in most years
        start = UTCDateTime(year=2015, month=12, day=31, hour=1)
        end = start + 24*3600
        years = np.arange(start.year, end.year+1)
        # days in start year
        startdays = np.arange(start.julday, 366)
        # Days in end year
        enddays = np.arange(1, end.julday+1)
        yeardirs = [os.path.join(
            self.outdir, str(year), self.net, self.stat) for year in years]
        startdaydirs = [os.path.join(
            yeardirs[0], 'HHE.D', "%s.%s.%s.%s.%s.%s.%s" %
            (self.net, self.stat, '00', 'HHE', 'D', years[0], day))
            for day in startdays]

        enddaydirs = [os.path.join(
            yeardirs[-1], 'HHE.D',
            "%s.%s.%s.%s.%s.%s.%s" %
            (self.net, self.stat, '00', 'HHE', 'D', years[-1], day))
            for day in enddays]
        glob_mock.side_effect = [yeardirs, startdaydirs, enddaydirs]
        #
        # Check if the directory is correct
        self.assertEqual(self.sc.sds_root, self.outdir)
        timetup = self.sc._get_times(self.net, self.stat)
        #
        t_control = (
            UTCDateTime(year=start.year, julday=start.julday),
            UTCDateTime(year=end.year, julday=end.julday)+24*3600)
        self.assertEqual(timetup, t_control)

    @mock.patch('seismic.trace_data.waveform.glob.glob')
    def test_get_available_stations(self, glob_mock):
        net = ['TOTAL']*3 + ['RANDOM']*3
        stat = ['RANDOM', 'BUT', 'SA', 'ME', 'LEN', 'GTH']
        glob_mock.return_value = [
            os.path.join(self.outdir, '1980', nc, sc) for nc, sc in zip(
                net, stat)]
        control_ret = [[nc, sc] for nc, sc in zip(net, stat)]
        self.assertEqual(control_ret, self.sc.get_available_stations())

    def test_no_available_data(self):
        with self.assertRaises(FileNotFoundError):
            self.sc.get_available_stations()


class TestLocalStoreClient(TestStoreClient):
    @mock.patch('seismic.trace_data.waveform.os.makedirs')
    @mock.patch('seismic.trace_data.waveform.os.path.isdir')
    @mock.patch('seismic.trace_data.waveform.os.listdir')
    @mock.patch.object(trace_data.waveform.Local_Store_Client,
                       "_set_inventory")
    def test_init(self, mock_setinv,
                  mock_listdir, mock_isdir, mock_makedirs):
        mock_isdir.return_value = True
        mock_setinv.sife_effect = lambda x: x.__setattr__("inventory",
                                                          Inventory())
        for rv in [True, False]:
            with self.subTest(rv=rv):
                mock_listdir.return_value = rv
                for m in [mock_setinv, mock_makedirs]:
                    m.reset_mock()

                test_config = deepcopy(config)
                sc = waveform.Local_Store_Client(test_config)

                self.assertEqual(
                    os.path.normpath(sc.sds_root), os.path.normpath(
                        os.path.join(test_config["proj_dir"],
                                     test_config["sds_dir"])))
                self.assertIsInstance(sc.rclient, waveform.sds.Client)
                self.assertIsInstance(sc.lclient, waveform.sds.Client)
                mock_makedirs.assert_called_once_with(test_config["proj_dir"],
                                                      exist_ok=True)
                mock_setinv.assert_called_once()
                self.assertTrue(hasattr(sc, "inventory"))

    @mock.patch('seismic.trace_data.waveform.os.makedirs')
    @mock.patch('seismic.trace_data.waveform.os.path.isdir')
    @mock.patch('seismic.trace_data.waveform.os.listdir')
    @mock.patch.object(trace_data.waveform.Local_Store_Client,
                       "_set_inventory")
    def test_init_without_default_paths(
            self, mock_setinv, mock_listdir, mock_isdir, mock_makedirs):
        mock_isdir.return_value = True
        mock_setinv.sife_effect = lambda x: x.__setattr__("inventory",
                                                          Inventory())
        for rv in [True, False]:
            with self.subTest(rv=rv):
                mock_listdir.return_value = rv
                for m in [mock_setinv, mock_makedirs]:
                    m.reset_mock()

                test_config = deepcopy(config)
                for k in ["sds_dir", "stationxml_file", "sds_fmtstr"]:
                    test_config.pop(k)

                sc = waveform.Local_Store_Client(test_config)

                self.assertEqual(
                    os.path.normpath(sc.sds_root), os.path.normpath(
                        os.path.join(test_config["proj_dir"],
                                     test_config["sds_dir"])))
                self.assertIsInstance(sc.rclient, waveform.sds.Client)
                self.assertIsInstance(sc.lclient, waveform.sds.Client)
                mock_makedirs.assert_called_once_with(test_config["proj_dir"],
                                                      exist_ok=True)
                mock_setinv.assert_called_once()
                self.assertTrue(hasattr(sc, "inventory"))
                self.assertTrue(all([
                    k in test_config for k in ["sds_dir", "stationxml_file",
                                               "sds_fmtstr"]]))

    @mock.patch('seismic.trace_data.waveform.os.makedirs')
    @mock.patch('seismic.trace_data.waveform.os.path.isdir')
    @mock.patch('seismic.trace_data.waveform.os.listdir')
    @mock.patch('seismic.trace_data.waveform.read_inventory')
    def test__set_inventory(self, mock_readinv,
                            mock_listdir, mock_isdir, mock_makedirs):
        mock_readinv.return_value = read_inventory()
        mock_isdir.return_value = True
        mock_listdir.return_value = True
        TEST_CONFIG = {"proj_dir": "test_proj_dir",
                       "net": {"network": ["GR"], "station": ["FUR", "WET"],
                               "component": "Z"},
                       "co": {"read_start": "2006-12-16 00:00:00",
                              "read_end": "2007-02-01 00:00:00"}
                       }

        for as_list in [True, False]:
            with self.subTest(as_list=as_list):
                test_config = deepcopy(TEST_CONFIG)
                if as_list:
                    test_config["net"]["network"] = ["GR"]
                    test_config["net"]["station"] = ["WET"]
                else:
                    test_config["net"]["network"] = "GR"
                    test_config["net"]["station"] = "WET"
                sc = waveform.Local_Store_Client(test_config)
                sc._set_inventory(test_config)
                mock_readinv.assert_called_with(test_config["stationxml_file"])
                self.assertIsInstance(sc.inventory, Inventory)
                self.assertEqual(len(sc.inventory), 0)

    @mock.patch('seismic.trace_data.waveform.os.makedirs')
    @mock.patch('seismic.trace_data.waveform.os.path.isdir')
    @mock.patch('seismic.trace_data.waveform.os.listdir')
    @mock.patch.object(trace_data.waveform.Local_Store_Client,
                       "_set_inventory")
    def test_read_inventory(
            self, mock_setinv, mock_listdir, mock_isdir, mock_makedirs):
        mock_isdir.return_value = True
        mock_listdir.return_value = True
        sc = waveform.Local_Store_Client(deepcopy(config))

        # Run if attribute `inventory` is not set
        delattr(sc, "inventory")
        inv = sc.read_inventory()
        self.assertIsInstance(inv, Inventory)
        self.assertEqual(len(inv), 0)

        # Run if `inventory` is set
        sc.__setattr__("inventory", read_inventory())
        inv = sc.read_inventory()
        self.assertIsInstance(inv, Inventory)
        self.assertEqual(len(inv), 2)


if __name__ == "__main__":
    unittest.main()
