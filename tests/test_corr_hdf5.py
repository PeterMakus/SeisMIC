'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 1st June 2021 10:42:03 am

Last Modified: Monday, 17th June 2024 05:09:04 pm

'''
from copy import deepcopy
import unittest
from unittest.mock import patch, MagicMock
from unittest import mock
import warnings

from obspy import read, UTCDateTime
from obspy.core import AttribDict
import numpy as np
import h5py
import yaml

from seismic.db import corr_hdf5
from seismic.correlate.stream import CorrStream, CorrTrace


with open('params_example.yaml') as file:
    co = corr_hdf5.co_to_hdf5(
        yaml.load(file, Loader=yaml.FullLoader)['co'])
    co['corr_args']['combinations'] = []


class TestConvertHeaderToHDF5(unittest.TestCase):
    def setUp(self):
        tr = read()[0]
        tr.decimate(4)  # so processing key is not empty
        self.stats = tr.stats

    def test_no_utc(self):
        # Check that all utcdatetime objects are now strings
        dataset = MagicMock()
        dataset.attrs = {}
        corr_hdf5.convert_header_to_hdf5(dataset, self.stats)
        for v in dataset.attrs.values():
            self.assertNotIsInstance(v, UTCDateTime)

    def test_length(self):
        # Check that all keys are transferred
        dataset = MagicMock()
        dataset.attrs = {}
        corr_hdf5.convert_header_to_hdf5(dataset, self.stats)
        self.assertEqual(dataset.attrs.keys(), self.stats.keys())


class TestReadHDF5Header(unittest.TestCase):
    def setUp(self) -> None:
        self.tr = read()[0]

    def test_result(self):
        dataset = MagicMock()
        dataset.attrs = {}
        self.tr.decimate(4)  # to put something into processing
        stats = self.tr.stats
        corr_hdf5.convert_header_to_hdf5(dataset, stats)
        self.assertEqual(corr_hdf5.read_hdf5_header(dataset), stats)

    def test_result_julday360(self):
        # There was a bug with that
        dataset = MagicMock()
        dataset.attrs = {}
        tr = read()[0]
        tr.decimate(4)  # to put something into processing
        stats = tr.stats
        self.tr.stats.starttime = UTCDateTime(
            year=2015, julday=360, hour=15, minute=3)
        corr_hdf5.convert_header_to_hdf5(dataset, stats)
        self.assertEqual(corr_hdf5.read_hdf5_header(dataset), stats)


def create_group_mock(d: dict, name: str, group: bool):
    """
    This is supposed to immitate the properties of
    :class:`h5py._hl.group.Group`

    :param d: dictionary
    :type d: dict
    :return: the mocked class
    :rtype: MagicMock
    """
    if group:
        m = MagicMock(spec=h5py._hl.group.Group)
    else:
        m = MagicMock()
    m.name = name
    m.__getitem__.side_effect = d.__getitem__
    m.__iter__.side_effect = d.__iter__
    m.__contains__.side_effect = d.__contains__
    m.values.side_effect = d.values
    return m


class TestAllTracesRecursive(unittest.TestCase):
    # The only thing I can do here is testing whether the conditions work
    @patch('seismic.db.corr_hdf5.read_hdf5_header')
    def test_is_np_array(self, read_header_mock):
        read_header_mock.return_value = None
        d = {
            'a': create_group_mock({}, '/outer_group/testname', False),
            'b': create_group_mock({}, '/outer_group/different_name', False)}

        g = create_group_mock(d, '/outer_group', True)
        st = CorrStream()
        st = corr_hdf5.all_traces_recursive(g, st, '/outer_group/testname')
        self.assertEqual(st.count(), 1)
        st = corr_hdf5.all_traces_recursive(
            g, st.clear(), '/outer_group/different_name')
        self.assertEqual(st.count(), 1)
        st = corr_hdf5.all_traces_recursive(g, st.clear(), '*name')
        self.assertEqual(st.count(), 2)
        st = corr_hdf5.all_traces_recursive(g, st.clear(), 'no_match')
        self.assertEqual(st.count(), 0)

    @patch('seismic.db.corr_hdf5.read_hdf5_header')
    def test_recursive(self, read_header_mock):
        # For this we need to patch fnmatch as well, as the names here aren't
        # full path
        read_header_mock.return_value = None
        d_innera = {
            'a': create_group_mock({}, '/outout/outer_group0/testname', False),
            'b': create_group_mock(
                {}, '/outout/outer_group0/different_name', False)}
        d_innerb = {
            'a': create_group_mock({}, '/outout/outer_group1/testname', False),
            'b': create_group_mock(
                {}, '/outout/outer_group1/different_name', False)}
        d_outer = {
            'A': create_group_mock(d_innera, '/outout/outer_group0', True),
            'B': create_group_mock(d_innerb, '/outout/outer_group1', True)}
        g = create_group_mock(d_outer, 'outout', True)
        st = CorrStream()
        st = corr_hdf5.all_traces_recursive(
            g, st, '/outout/outer_group0/testname')
        self.assertEqual(st.count(), 1)
        st = corr_hdf5.all_traces_recursive(g, st.clear(), '*')
        self.assertEqual(st.count(), 4)


class TestDBHandler(unittest.TestCase):
    @patch('seismic.db.corr_hdf5.h5py.File.__init__')
    def setUp(self, super_mock):
        self.file_mock = MagicMock()
        super_mock.return_value = self.file_mock
        self.dbh = corr_hdf5.DBHandler('a', 'r', 'gzip9', None, False)
        tr = read()[0]
        tr.data = np.ones_like(tr.data, dtype=int)
        tr.stats['corr_start'] = tr.stats.starttime
        tr.stats['corr_end'] = tr.stats.endtime
        self.ctr = CorrTrace(tr.data, _header=tr.stats)

    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_compression_indentifier(self, getitem_mock):
        d = {'test': 0}
        getitem_mock.side_effect = d.__getitem__
        self.assertEqual(self.dbh.compression, 'gzip')
        self.assertEqual(self.dbh.compression_opts, 9)
        self.assertEqual(self.dbh['test'], 0)

    @patch('seismic.db.corr_hdf5.super')
    def test_forbidden_compression(self, super_mock):
        super_mock.return_value = None
        with self.assertRaises(ValueError):
            _ = corr_hdf5.DBHandler('a', 'a', 'notexisting5', None, False)

    @patch('seismic.db.corr_hdf5.super')
    def test_forbidden_compression_level(self, super_mock):
        super_mock.return_value = None
        with warnings.catch_warnings(record=True) as w:
            dbh = corr_hdf5.DBHandler('a', 'a', 'gzip10', None, False)
            self.assertEqual(dbh.compression_opts, 9)
            self.assertEqual(len(w), 1)

    @patch('seismic.db.corr_hdf5.super')
    def test_no_compression_level(self, super_mock):
        super_mock.return_value = None
        with self.assertRaises(IndexError):
            _ = corr_hdf5.DBHandler('a', 'a', 'gzip', None, False)

    @patch('seismic.db.corr_hdf5.super')
    def test_no_compression_name(self, super_mock):
        super_mock.return_value = None
        with self.assertRaises(IndexError):
            _ = corr_hdf5.DBHandler('a', 'a', '9', None, False)

    def test_add_already_available_data(self):
        st = self.ctr.stats
        path = corr_hdf5.hierarchy.format(
            tag='subdivision',
            network=st.network, station=st.station, channel=st.channel,
            location=st.location,
            corr_st=st.corr_start.format_fissures(),
            corr_et=st.corr_end.format_fissures())
        with warnings.catch_warnings(record=True) as w:
            with patch.object(self.dbh, 'create_dataset') as create_ds_mock:
                create_ds_mock.side_effect = ValueError('test')
                self.dbh.add_correlation(self.ctr)
                create_ds_mock.assert_called_with(
                    path, data=self.ctr.data, compression='gzip',
                    compression_opts=9)
            self.assertEqual(len(w), 1)

    @patch('seismic.db.corr_hdf5.super')
    def test_add_different_object(self, super_mock):
        super_mock.return_value = None
        dbh = corr_hdf5.DBHandler('a', 'r', 'gzip9', None, False)
        with self.assertRaises(TypeError):
            dbh.add_correlation(read())

    @patch('seismic.db.corr_hdf5.read_hdf5_header')
    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_data_no_wildcard(self, file_mock, read_hdf5_header_mock):
        read_hdf5_header_mock.return_value = self.ctr.stats
        net = 'AB-CD'
        stat = 'AB-CD'
        ch = 'XX-XX'
        tag = 'rand'
        loc = '00'
        corr_start = UTCDateTime(0)
        corr_end = UTCDateTime(100)
        exp_path = corr_hdf5.hierarchy.format(
            tag=tag, network=net, station=stat, channel=ch,
            location=loc,
            corr_st=corr_start.format_fissures(),
            corr_et=corr_end.format_fissures())
        d = {exp_path: self.ctr.data}
        file_mock.side_effect = d.__getitem__
        self.assertTrue(np.all(self.dbh[exp_path] == d[exp_path]))
        outdata = self.dbh.get_data(
            net, stat, loc, ch, tag, corr_start, corr_end)
        self.assertEqual(outdata[0], self.ctr)
        file_mock.assert_called_with(exp_path)

    @patch('seismic.db.corr_hdf5.read_hdf5_header')
    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_data_no_wildcard_not_alphabetical(
            self, file_mock, read_hdf5_header_mock):
        read_hdf5_header_mock.return_value = self.ctr.stats
        net = 'CD-AB'
        stat = 'AB-CD'
        ch = 'XX-XX'
        loc = '00-00'
        tag = 'rand'
        corr_start = UTCDateTime(0)
        corr_end = UTCDateTime(100)
        exp_path = corr_hdf5.hierarchy.format(
            tag=tag, network='AB-CD', station='CD-AB',
            location=loc, channel=ch,
            corr_st=corr_start.format_fissures(),
            corr_et=corr_end.format_fissures())
        d = {exp_path: self.ctr.data}
        file_mock.side_effect = d.__getitem__
        out = self.dbh.get_data(
            net, stat, loc, ch, tag, corr_start, corr_end)
        file_mock.assert_called_with(exp_path)
        self.assertEqual(out[0], self.ctr)

    @patch('seismic.db.corr_hdf5.all_traces_recursive')
    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_data_wildcard(self, file_mock, all_tr_recursive_mock):
        all_tr_recursive_mock.return_value = None
        net = 'AB-CD'
        stat = '*'
        ch = '*'
        loc = '*'
        tag = 'rand'
        corr_start = UTCDateTime(0)
        corr_end = UTCDateTime(100)
        exp_path = corr_hdf5.hierarchy.format(
            tag=tag, network=net, station=stat, channel=ch,
            location=loc,
            corr_st=corr_start.format_fissures(),
            corr_et=corr_end.format_fissures())
        d = {exp_path: self.ctr.data, '/rand/AB-CD/': self.ctr.data}
        file_mock.side_effect = d.__getitem__

        _ = self.dbh.get_data(net, stat, loc, ch, tag, corr_start, corr_end)
        file_mock.assert_called_with('/rand/AB-CD/')
        all_tr_recursive_mock.assert_called_with(
            d['/rand/AB-CD/'], CorrStream(), '/rand/AB-CD***/%s/%s' % (
                corr_start.format_fissures(), corr_end.format_fissures()))

    @patch('seismic.db.corr_hdf5.all_traces_recursive')
    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_data_wildcard2(self, file_mock, all_tr_recursive_mock):
        all_tr_recursive_mock.return_value = None
        net = 'AB-CD'
        stat = '*'
        ch = '*'
        loc = '*'
        tag = 'rand'
        corr_start = '*'
        corr_end = '*'
        exp_path = corr_hdf5.hierarchy.format(
            tag=tag, network=net, station=stat, channel=ch, location=loc,
            corr_st=corr_start, corr_et=corr_end)
        exp_path = '/'.join(exp_path.split('/')[:-4])
        d = {exp_path: self.ctr.data, '/rand/AB-CD/': self.ctr.data}
        file_mock.side_effect = d.__getitem__

        _ = self.dbh.get_data(net, stat, ch, loc, tag, corr_start, corr_end)
        file_mock.assert_called_with('/rand/AB-CD/')
        all_tr_recursive_mock.assert_called_with(
            d['/rand/AB-CD/'], CorrStream(), '/rand/AB-CD*****')

    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_available_starttimes(self, file_mock):
        net = 'AB-CD'
        stat = 'XY-YZ'
        ch = 'HHZ'
        loc = '00'
        tag = 'tag'
        d = {}
        starttimes = {
            UTCDateTime(0).format_fissures(): None,
            UTCDateTime(200).format_fissures(): None}
        d['/'+'/'.join([tag, net, stat, loc, ch])] = starttimes
        file_mock.side_effect = d.__getitem__
        exp_result = {ch: list(starttimes.keys())}
        self.assertEqual(
            self.dbh.get_available_starttimes(net, stat, tag, loc, ch),
            exp_result)
        file_mock.assert_called_with('/'+'/'.join([tag, net, stat, loc, ch]))

    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_available_starttimes_key_error(self, file_mock):
        net = 'AB-CD'
        stat = 'XY-YZ'
        loc = '00'
        ch = 'HHZ'
        tag = 'tag'
        d = {}
        starttimes = {
            UTCDateTime(0).format_fissures(): None,
            UTCDateTime(200).format_fissures(): None}
        d['/'+'/'.join([tag, net, stat, loc, ch])] = starttimes
        file_mock.side_effect = d.__getitem__
        self.assertEqual(
            self.dbh.get_available_starttimes(net, stat, tag, loc, 'blub'),
            {})

    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_available_starttimes_wildcard(self, file_mock):
        net = 'AB-CD'
        stat = 'XY-YZ'
        loc = '00'
        ch = 'HHZ'
        tag = 'tag'
        d = {}
        starttimes = {
            UTCDateTime(0).format_fissures(): None,
            UTCDateTime(200).format_fissures(): None}
        d['/'+'/'.join([tag, net, stat, loc])] = {}
        d['/'+'/'.join([tag, net, stat, loc])][ch] = starttimes
        d['/'+'/'.join([tag, net, stat, loc, ch])] = starttimes
        file_mock.side_effect = d.__getitem__
        exp_result = {ch: list(starttimes)}
        self.assertEqual(
            self.dbh.get_available_starttimes(net, stat, tag, loc, '*'),
            exp_result)

    @patch('seismic.db.corr_hdf5.DBHandler.get_corr_options')
    @patch('seismic.db.corr_hdf5.h5py.File.__init__')
    def test_wrong_co(self, super_mock, gco_mock):
        self.file_mock = MagicMock()
        super_mock.return_value = self.file_mock
        oco = deepcopy(co)
        oco['sampling_rate'] = 100000
        gco_mock.return_value = oco
        with self.assertRaises(PermissionError):
            corr_hdf5.DBHandler('a', 'a', 'gzip9', co, False)

    @patch('seismic.db.corr_hdf5.DBHandler.get_corr_options')
    @patch('seismic.db.corr_hdf5.h5py.File.__init__')
    def test_wrong_co2(self, super_mock, gco_mock):
        self.file_mock = MagicMock()
        super_mock.return_value = self.file_mock
        oco = deepcopy(co)
        oco['nlub'] = 100000
        gco_mock.return_value = oco
        with self.assertRaises(PermissionError):
            corr_hdf5.DBHandler('a', 'a', 'gzip9', co, False)

    @patch('seismic.db.corr_hdf5.DBHandler.get_corr_options')
    @patch('seismic.db.corr_hdf5.h5py.File.__init__')
    def test_wrong_co_force(self, super_mock, gco_mock):
        self.file_mock = MagicMock()
        super_mock.return_value = self.file_mock
        oco = deepcopy(co)
        oco['nlub'] = 100000
        gco_mock.return_value = oco
        corr_hdf5.DBHandler('a', 'a', 'gzip9', co, True)

    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_corr_options(self, gi_mock):
        d = {'co': AttribDict(attrs={'co': str(corr_hdf5.co_to_hdf5(co))})}
        gi_mock.side_effect = d.__getitem__
        self.assertEqual(corr_hdf5.co_to_hdf5(co), self.dbh.get_corr_options())

    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_corr_options_no_data(self, gi_mock):
        d = {}
        gi_mock.side_effect = d.__getitem__
        with self.assertRaises(KeyError):
            self.dbh.get_corr_options()

    @patch('seismic.db.corr_hdf5.h5py.File.create_dataset')
    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_add_corr_options(self, gi_mock, cd_mock):
        d = {'co': AttribDict(attrs={})}
        gi_mock.side_effect = d.__getitem__
        cd_mock.return_value = d['co']
        self.dbh.add_corr_options(co)
        self.assertEqual(corr_hdf5.co_to_hdf5(co), self.dbh.get_corr_options())

    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_available_channels(self, gi_mock):
        net = 'mynet'
        stat = 'mystat'
        loc = 'myloc'
        tag = 'bla'
        path = '/%s/' % '/'.join([tag, net, stat, loc])
        d = {path: {'a': 0, 'b': 1, 'c': 2}}
        gi_mock.side_effect = d.__getitem__
        exp = ['a', 'b', 'c']
        self.assertEqual(exp, self.dbh.get_available_channels(
            tag, net, stat, loc))

    @patch('seismic.db.corr_hdf5.h5py.File.__getitem__')
    def test_get_available_channels_none_available(self, gi_mock):
        net = 'mynet'
        stat = 'mystat'
        loc = 'myloc'
        tag = 'bla'
        d = {}
        gi_mock.side_effect = d.__getitem__
        self.assertEqual([], self.dbh.get_available_channels(
            tag, net, stat, loc))

    def test_remove_data_wildcard_network(self):
        with self.assertRaises(ValueError):
            self.dbh.remove_data('*', 'x', 'x', 'x', 'x', 'x')

    def test_remove_data_wildcard_station(self):
        with self.assertRaises(ValueError):
            self.dbh.remove_data('x', '*', 'x', 'x', 'x', 'x')

    def test_remove_data_wildcard_channel(self):
        with self.assertRaises(ValueError):
            self.dbh.remove_data('x', 'x', 'x', '*', 'x', 'x')

    def test_remove_data_corrstart_wrong_type(self):
        with self.assertRaises(TypeError):
            self.dbh.remove_data('x', 'x', 'x', 'x', 'x', 3)

    @patch('seismic.db.corr_hdf5.h5py.File.__delitem__')
    def test_remove_data_no_wildcard_not_alphabetical(
            self, file_mock):
        net = 'CD-AB'
        stat = 'AB-CD'
        ch = 'XX-XX'
        tag = 'rand'
        loc = '00-00'
        corr_start = UTCDateTime(0)
        exp_path = corr_hdf5.hierarchy.format(
            tag=tag, network='AB-CD', station='CD-AB', location=loc,
            channel=ch,
            corr_st=corr_start.format_fissures(),
            corr_et='')[:-1]
        d = {exp_path: 'x'}
        file_mock.side_effect = d.__delitem__
        self.dbh.remove_data(net, stat, loc, ch, tag, corr_start)
        file_mock.assert_called_with(exp_path)
        self.assertDictEqual(d, {})

    @patch('seismic.db.corr_hdf5.h5py.File.__delitem__')
    def test_remove_data_no_wildcard(
            self, file_mock):
        net = 'AB-CD'
        stat = 'CD-AB'
        ch = 'XX-XX'
        loc = '00-00'
        tag = 'rand'
        corr_start = UTCDateTime(0)
        exp_path = corr_hdf5.hierarchy.format(
            tag=tag, network='AB-CD', station='CD-AB', channel=ch,
            location=loc,
            corr_st=corr_start.format_fissures(),
            corr_et='')[:-1]
        d = {exp_path: 'x'}
        file_mock.side_effect = d.__delitem__
        self.dbh.remove_data(net, stat, loc, ch, tag, corr_start)
        file_mock.assert_called_with(exp_path)
        self.assertDictEqual(d, {})

    @patch('seismic.db.corr_hdf5.h5py.File.__delitem__')
    def test_remove_data_not_found(
            self, file_mock):
        net = 'AB-CD'
        stat = 'CD-AB'
        ch = 'XX-XX'
        loc = '00-00'
        tag = 'rand'
        corr_start = UTCDateTime(0)
        exp_path = corr_hdf5.hierarchy.format(
            tag=tag, network='AB-CD', station='CD-AB', channel=ch,
            location=loc,
            corr_st=corr_start.format_fissures(),
            corr_et='')[:-1]
        act_path = corr_hdf5.hierarchy.format(
            tag=tag, network='AB-CD', station='OT-AB', channel=ch,
            location=loc,
            corr_st=corr_start.format_fissures(),
            corr_et='')[:-1]
        d = {act_path: 'x'}
        file_mock.side_effect = d.__delitem__
        with warnings.catch_warnings(record=True) as w:
            self.dbh.remove_data(net, stat, loc, ch, tag, corr_start)
            file_mock.assert_called_with(exp_path)
            self.assertEqual(len(w), 1)
        self.assertDictEqual(d, {act_path: 'x'})

    @patch('seismic.db.corr_hdf5.h5py.File.__delitem__')
    def test_remove_data_corrstart_wildcard(
            self, file_mock):
        net = 'AB-CD'
        stat = 'CD-AB'
        ch = 'XX-XX'
        loc = '00-00'
        tag = 'rand'
        corr_start = '*'
        exp_path = corr_hdf5.hierarchy.format(
            tag=tag, network='AB-CD', station='CD-AB', channel=ch,
            location=loc,
            corr_st='*', corr_et='')[:-3]
        act_path = corr_hdf5.hierarchy.format(
            tag=tag, network='AB-CD', station='CD-AB', channel=ch,
            location=loc,
            corr_st=UTCDateTime(0).format_fissures(),
            corr_et='bla')
        d = {act_path: 'x'}
        file_mock.side_effect = d.__delitem__
        self.dbh.remove_data(net, stat, loc, ch, tag, corr_start)
        file_mock.assert_called_with(exp_path)

    @patch('seismic.db.corr_hdf5.h5py.File.__delitem__')
    def test_remove_data_partial_wildcard(self, file_mock):
        net = 'AB-CD'
        stat = 'CD-AB'
        ch = 'XX-XX'
        tag = 'rand'
        loc = '00-00'
        # technically deletes the first 9 days of 1970
        corr_start = UTCDateTime(0).format_fissures()[:-14] + '*'
        act_paths = [
            corr_hdf5.hierarchy.format(
                tag=tag, network=net, station=stat, channel=ch,
                location=loc,
                corr_st=(UTCDateTime(0) + x).format_fissures(),
                corr_et='')[:-1] for x in np.arange(11)*86400]
        d = {x: 'x' for x in act_paths}
        file_mock.side_effect = d.__delitem__
        with mock.patch.object(
                self.dbh, 'get_available_starttimes') as gas_mock:
            gas_mock.return_value = {ch: [
                (UTCDateTime(0) + x).format_fissures()
                for x in np.arange(11)*86400]}
            self.dbh.remove_data(net, stat, loc, ch, tag, corr_start)
            gas_mock.assert_called_once_with(
                net, stat, tag, loc, ch)
        for x in act_paths[:-2]:
            file_mock.assert_any_call(x)
        # the last two times do not fit the pattern
        self.assertDictEqual(
            d, {x: 'x' for x in act_paths[-2:]})


class TestCorrelationDataBase(unittest.TestCase):
    @patch('seismic.db.corr_hdf5.DBHandler')
    def test_no_corr_options(self, dbh_mock):
        with warnings.catch_warnings(record=True) as w:
            cdb = corr_hdf5.CorrelationDataBase('a', None, 'a')
        self.assertEqual(cdb.mode, 'r')
        self.assertEqual(len(w), 1)

    @patch('seismic.db.corr_hdf5.DBHandler')
    def test_path_name(self, dbh_mock):
        cdb = corr_hdf5.CorrelationDataBase('a', None, 'r')
        self.assertEqual(cdb.path, 'a.h5')


class TestCoToHDF5(unittest.TestCase):
    def test_pop_keys(self):
        d = {
            'subdir': 0, 'starttime': 15, 'corr_args': {'combinations': 3},
            'subdivision': {
                'recombine_subdivision': False,
                'delete_subdivision': True},
            'preProcessing': [
                {'function': 'function1', 'args': 'bla'},
                {'function': 'myimport.stream_mask_at_utc'}
            ]}
        coc = corr_hdf5.co_to_hdf5(d)
        self.assertDictEqual(coc, {
            'corr_args': {}, 'subdivision': {},
            'preProcessing': [{'function': 'function1', 'args': 'bla'}]})
        # Make sure that input is not altered
        self.assertDictEqual(d, {
            'subdir': 0, 'starttime': 15, 'corr_args': {'combinations': 3},
            'subdivision': {
                'recombine_subdivision': False,
                'delete_subdivision': True},
            'preProcessing': [
                {'function': 'function1', 'args': 'bla'},
                {'function': 'myimport.stream_mask_at_utc'}
            ]})

    def test_keyError_handling(self):
        d = {'corr_args': {'a': 1}, 'subdivision': {'bla': 'g'}}
        coc = corr_hdf5.co_to_hdf5(d)
        self.assertDictEqual(d, coc)


if __name__ == "__main__":
    unittest.main()
