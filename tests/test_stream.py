'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 31st May 2021 01:50:04 pm
Last Modified: Monday, 18th July 2022 10:33:30 am
'''

from multiprocessing.sharedctypes import Value
import unittest
from unittest import mock
import warnings
from copy import deepcopy

import numpy as np
from obspy import read, read_inventory, Stream
from obspy.core.utcdatetime import UTCDateTime

from seismic.correlate import stream


class TestCorrBulk(unittest.TestCase):
    def setUp(self):
        st = read()
        A = np.zeros((st.count(), st[0].stats.npts))
        statl = []
        keys = [
            'stla', 'stlo', 'stel', 'evla', 'evlo', 'evel', 'dist',
            'baz', 'az']
        for k in keys:
            st[0].stats[k] = 0
        statso = stream.CorrStats(st[0].stats)
        for ii, tr in enumerate(st):
            stats = deepcopy(statso)
            stats.corr_start += 24*3600*ii
            stats.corr_end = stats.corr_start + stats.npts*stats.delta
            A[ii] = tr.data
            statl.append(stats)
        self.cb = stream.CorrBulk(A, statlist=statl)
        np.testing.assert_array_equal(A, self.cb.data)

    def test_setup_no_stats(self):
        cb = stream.CorrBulk(np.zeros((5, 5)))
        self.assertEqual(cb.stats.ntrcs, 5)
        self.assertEqual(cb.stats.npts, 5)
        self.assertEqual(cb.stats.processing_bulk, [])
        self.assertIsNone(cb.ref_trc)
        np.testing.assert_array_equal(np.zeros((5, 5)), cb.data)

    def test_setup_ex_stats(self):
        cb = stream.CorrBulk(np.zeros((5, 5)), stats=deepcopy(self.cb.stats))
        for v0, v1 in zip(cb.stats.values(), self.cb.stats.values()):
            self.assertEqual(v0, v1)
        np.testing.assert_array_equal(cb.data, np.zeros((5, 5)))

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_normalize')
    def test_normalize(self, cmn_mock):
        cb = self.cb.copy()
        cb.normalize(0, 0, 'blatest')
        cmn_mock.assert_called_once_with(
            mock.ANY, cb.stats, 0, 0, 'blatest')
        self.assertIn(
            'normalize; normtype: blatest, starttime: 0, endtime: 0',
            cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_clip')
    def test_clip(self, clip_mock):
        cb = self.cb.copy()
        clip_mock.return_value = np.zeros((5,))
        cb.clip(0, None)
        clip_mock.assert_called_once_with(
            mock.ANY, 0, None)
        np.testing.assert_array_equal(cb.data, np.zeros((5,)))
        self.assertIn(
            'Clipped; threshold: 0*std, axis=None',
            cb.stats.processing_bulk)

    def test_copy(self):
        cb = self.cb.copy()
        cb.data += 25
        cb.stats['eg'] = 1
        np.testing.assert_array_almost_equal(
            cb.data-self.cb.data, 25*np.ones_like(cb.data))
        with self.assertRaises(KeyError):
            print(self.cb.stats['eg'])

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_correct_decay')
    def test_correct_decay(self, decay_mock):
        # Actually just need to feed in stuff. Algorithm is tested elsewhere
        cb = self.cb.copy()
        decay_mock.return_value = np.zeros((25, 25))
        cb.correct_decay()
        decay_mock.assert_called_once_with(mock.ANY, cb.stats)
        np.testing.assert_array_equal(decay_mock.call_args[0][0], self.cb.data)
        np.testing.assert_array_equal(np.zeros((25, 25)), cb.data)
        self.assertIn(
            'Corrected for Amplitude Decay', cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.time_shift_apply')
    def test_correct_shift(self, shift_mock):
        cb = self.cb.copy()
        shift_mock.return_value = np.zeros((25, 25))
        dvmock = mock.MagicMock()
        dvmock.value = 1
        dvmock.value_type = 'shift'
        cb.correct_shift(dv=dvmock)
        shift_mock.assert_called_once_with(
            mock.ANY, dvmock.value, single_sided=False)
        np.testing.assert_array_equal(
            shift_mock.call_args[0][0], self.cb.data)
        np.testing.assert_array_equal(np.zeros((25, 25)), cb.data)
        self.assertIn(
            'Applied time shift', cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.time_shift_apply')
    def test_correct_shift2(self, shift_mock):
        cb = self.cb.copy()
        shift_mock.return_value = np.zeros((25, 25))
        value = 1
        cb.correct_shift(shift=value)
        shift_mock.assert_called_once_with(mock.ANY, value, single_sided=False)
        np.testing.assert_array_equal(
            shift_mock.call_args[0][0], self.cb.data)
        np.testing.assert_array_equal(np.zeros((25, 25)), cb.data)
        self.assertIn(
            'Applied time shift', cb.stats.processing_bulk)

    def test_correct_shift_wrong_type(self):
        dvmock = mock.MagicMock()
        dvmock.value = 1
        dvmock.value_type = 'bla'
        with self.assertRaises(ValueError):
            self.cb.correct_shift(dv=dvmock)

    def test_correct_shift_no_args(self):
        with self.assertRaises(ValueError):
            self.cb.correct_shift()

    def test_correct_shift_two_args(self):
        with self.assertRaises(ValueError):
            self.cb.correct_shift(dv='bla', shift='blub')

    @mock.patch('seismic.correlate.stream.time_stretch_apply')
    def test_correct_stretch(self, stretch_mock):
        cb = self.cb.copy()
        stretch_mock.return_value = np.zeros((25, 25))
        dvmock = mock.MagicMock()
        dvmock.value = 1
        dvmock.value_type = 'stretch'
        cb.correct_stretch(dvmock)
        stretch_mock.assert_called_once_with(mock.ANY, -1.*dvmock.value, False)
        np.testing.assert_array_equal(
            stretch_mock.call_args[0][0], self.cb.data)
        np.testing.assert_array_equal(np.zeros((25, 25)), cb.data)
        self.assertIn(
            'Applied time stretch', cb.stats.processing_bulk)

    def test_correct_stretch_wrong_type(self):
        dvmock = mock.MagicMock()
        dvmock.value = 1
        dvmock.value_type = 'bla'
        with self.assertRaises(ValueError):
            self.cb.correct_stretch(dvmock)

    def test_create_corrstream(self):
        # Lets's just create a CorrStream from stretch and then convert back
        # and forth
        st = read()
        for tr in st:
            # Else this will need too much time
            tr.stats.sampling_rate = 1
        cst = stream.CorrStream()
        for tr in st:
            delta = tr.stats.endtime - tr.stats.starttime
            tr.data = np.ones_like(tr.data)
            tr.stats['corr_start'] = tr.stats.starttime
            tr.stats['corr_end'] = tr.stats.endtime
            tr.stats['stla'] = 0
            tr.stats['stlo'] = 0
            tr.stats['stel'] = 0
            tr.stats['evla'] = 0
            tr.stats['evlo'] = 0
            tr.stats['evel'] = 0
            tr.stats['dist'] = 0
            tr.stats['baz'] = 0
            tr.stats['az'] = 0
            tr.stats['channel'] = 'HHE'
            tr = stream.CorrTrace(tr.data, _header=tr.stats)
            for ii in range(10):
                ntr = tr.copy()
                ntr.data = np.empty(tr.data.shape)
                ntr.stats['corr_start'] += delta*(ii+1)
                ntr.stats['corr_end'] += delta*(ii+1)
                cst.append(ntr)
        cb = cst.create_corr_bulk(inplace=False)
        cst2 = cb.create_corr_stream()
        for tr0, tr in zip(cst, cst2):
            np.testing.assert_array_equal(tr0.data, tr.data)
            for k in tr0.stats.keys():
                self.assertEqual(tr0.stats[k], tr.stats[k])

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_envelope')
    def test_envelope(self, envelope_mock):
        envelope_mock.return_value = np.zeros((25, 25))
        cb = self.cb.copy()
        cb.envelope()
        np.testing.assert_array_equal(
            envelope_mock.call_args[0][0], self.cb.data)
        np.testing.assert_array_equal(np.zeros((25, 25)), cb.data)
        self.assertIn('Computed Envelope', cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_filter')
    def test_filter(self, filter_mock):
        filter_mock.return_value = np.zeros((25, 25))
        cb = self.cb.copy()
        cb.filter((1, 2), 17)
        filter_mock.assert_called_once_with(mock.ANY, cb.stats, (1, 2), 17)
        np.testing.assert_array_equal(
            filter_mock.call_args[0][0], self.cb.data)
        self.assertIn(
            'filter; freqs: (1, 2), order: 17', cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_extract_trace')
    def test_extract_trace(self, extract_mock):
        extract_mock.return_value = np.zeros((25,))
        cb = self.cb.copy()
        out = cb.extract_trace('bla', 25)
        extract_mock.assert_called_once_with(mock.ANY, cb.stats, 'bla', 25)
        np.testing.assert_array_equal(
            extract_mock.call_args[0][0], self.cb.data)
        np.testing.assert_array_equal(out, np.zeros((25,)))
        np.testing.assert_array_equal(cb.ref_trc, np.zeros((25,)))

    def test_extract_multi_trace_single(self):
        with mock.patch.object(self.cb, 'extract_trace') as extract_mock:
            extract_mock.return_value = 'test'
            t = self.cb.extract_multi_trace(0, 'bla', 25)
            extract_mock.assert_called_once_with('bla', 25)
        self.assertEqual(t, 'test')

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_extract_trace')
    def test_extract_multi_trace_win_inc_int(self, extract_mock):
        extract_mock.return_value = np.zeros((25,))
        wi = 1  # day
        rtrcs = self.cb.extract_multi_trace(wi, 'bla', 25)
        self.assertEqual(len(rtrcs), 3)
        np.testing.assert_array_equal(np.zeros((3, 25)), rtrcs)
        calls = [mock.call(mock.ANY, self.cb.stats, 'bla', 25)]*3
        extract_mock.assert_has_calls(calls)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_extract_trace')
    def test_extract_multi_trace_win_inc_vect(self, extract_mock):
        extract_mock.return_value = np.zeros((25,))
        wi = [0.5, 0.5, 1, 0.5, 0.5]  # day
        rtrcs = self.cb.extract_multi_trace(wi, 'bla', 25)
        self.assertEqual(len(rtrcs), 5)
        np.testing.assert_array_equal(np.zeros((5, 25)), rtrcs)
        calls = [mock.call(mock.ANY, self.cb.stats, 'bla', 25)]*5
        extract_mock.assert_has_calls(calls)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_mirror')
    def test_mirror(self, mirror_mock):
        cb = self.cb.copy()
        mirror_mock.return_value = (np.zeros((25, 25)), cb.stats)
        cb.mirror()
        mirror_mock.assert_called_once_with(mock.ANY, cb.stats)
        np.testing.assert_array_equal(
            mirror_mock.call_args[0][0], self.cb.data)
        np.testing.assert_array_equal(cb.data, np.zeros((25, 25)))
        self.assertIn('Mirrored.', cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_resample')
    def test_resample(self, resample_mock):
        cb = self.cb.copy()
        resample_mock.return_value = (np.zeros((25, 25)), cb.stats)
        cb.resample([1, 2, 3], [4, 5, 6])
        resample_mock.assert_called_once_with(
            mock.ANY, cb.stats, [1, 2, 3], [4, 5, 6])
        np.testing.assert_array_equal(
            resample_mock.call_args[0][0], self.cb.data)
        np.testing.assert_array_equal(cb.data, np.zeros((25, 25)))
        self.assertIn(
            'Resampled. Starttimes: [1, 2, 3], Endtimes: [4, 5, 6]',
            cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_resample_or_decimate')
    def test_resample_time_axis(self, rd_mock):
        cb = self.cb.copy()
        rd_mock.return_value = (np.zeros((25, 25)), cb.stats)
        cb.resample_time_axis(25)
        rd_mock.assert_called_once_with(
            mock.ANY, cb.stats, 25)
        np.testing.assert_array_equal(
            rd_mock.call_args[0][0], self.cb.data)
        np.testing.assert_array_equal(cb.data, np.zeros((25, 25)))
        self.assertIn(
            'Resampled time axis. New sampling rate: 25Hz',
            cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_smooth')
    def test_smooth(self, smooth_mock):
        cb = self.cb.copy()
        smooth_mock.return_value = np.zeros((25, 25))
        cb.smooth(1, 'blub', 125)
        smooth_mock.assert_called_once_with(mock.ANY, 1, 'blub', 125)
        np.testing.assert_array_equal(
            smooth_mock.call_args[0][0], self.cb.data)
        np.testing.assert_array_equal(cb.data, np.zeros((25, 25)))
        self.assertIn(
            'Smoothed. wsize: 1, wtype: blub, axis: 125',
            cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.DV')
    @mock.patch('seismic.correlate.stream.pcp.corr_mat_shift')
    def test_shift(self, shift_mock, dv_mock):
        shift_mock.return_value = {'test': 0}
        self.cb.find_clock_shift(
            np.zeros((25,)), [1, 2, 3], 0.5, 105, 'bla', True)
        shift_mock.assert_called_once_with(
            mock.ANY, self.cb.stats, mock.ANY, [1, 2, 3], 0.5, 105, 'bla',
            True)
        np.testing.assert_array_equal(
            shift_mock.call_args[0][2], np.zeros((25,)))
        dv_mock.assert_called_once_with(test=0)

    @mock.patch('seismic.correlate.stream.DV')
    @mock.patch('seismic.correlate.stream.pcp.corr_mat_shift')
    def test_shift2(self, shift_mock, dv_mock):
        shift_mock.return_value = {'test': 0}
        self.cb.ref_trc = 'ha_funny!'
        self.cb.find_clock_shift()
        shift_mock.assert_called_once_with(
            mock.ANY, self.cb.stats, 'ha_funny!', None, 10, 101, 'both',
            False)
        np.testing.assert_array_equal(
            shift_mock.call_args[0][0], self.cb.data)
        dv_mock.assert_called_once_with(test=0, sim_mat=mock.ANY)

    @mock.patch('seismic.correlate.stream.DV')
    @mock.patch('seismic.correlate.stream.pcp.corr_mat_stretch')
    def test_stretch(self, stretch_mock, dv_mock):
        stretch_mock.return_value = {'test': 0}
        self.cb.stretch(np.zeros((25,)), [1, 2, 3], 0.5, 105, 'bla', True)
        stretch_mock.assert_called_once_with(
            mock.ANY, self.cb.stats, mock.ANY, [1, 2, 3], 0.5, 105, 'bla',
            True)
        np.testing.assert_array_equal(
            stretch_mock.call_args[0][2], np.zeros((25,)))
        dv_mock.assert_called_once_with(test=0)

    @mock.patch('seismic.correlate.stream.DV')
    @mock.patch('seismic.correlate.stream.pcp.corr_mat_stretch')
    def test_stretch2(self, stretch_mock, dv_mock):
        stretch_mock.return_value = {'test': 0}
        self.cb.ref_trc = 'ha_funny!'
        self.cb.stretch()
        stretch_mock.assert_called_once_with(
            mock.ANY, self.cb.stats, 'ha_funny!', None, 0.1, 101, 'both',
            False)
        np.testing.assert_array_equal(
            stretch_mock.call_args[0][0], self.cb.data)
        dv_mock.assert_called_once_with(test=0, sim_mat=mock.ANY)

    @mock.patch('seismic.correlate.stream.m3ut.save_header_to_np_array')
    @mock.patch('seismic.correlate.stream.np.savez_compressed')
    def test_save(self, np_save_mock, save_header_mock):
        save_header_mock.return_value = {'test': 0}
        self.cb.save('mypath')
        save_header_mock.assert_called_once_with(self.cb.stats)
        np_save_mock.assert_called_once_with('mypath', data=mock.ANY, test=0)

    def test_slice(self):
        with mock.patch.object(self.cb, '_find_slice_index') as fsi_mock:
            fsi_mock.return_value = np.array([False, True, False])
            cbsl = self.cb.slice('start', 'stop', True)
            fsi_mock.assert_called_once_with('start', 'stop', True)
        np.testing.assert_array_equal(
            self.cb.data[[False, True, False], :], cbsl.data)
        for vsl, (k, v) in zip(cbsl.stats.values(), self.cb.stats.items()):
            if isinstance(v, list) and k != 'processing_bulk':
                self.assertListEqual([v[1]], vsl)
            elif isinstance(v, np.ndarray):
                np.testing.assert_array_equal(vsl, v[1])
            else:
                self.assertEqual(v, vsl)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_taper')
    def test_taper(self, taper_mock):
        cb = self.cb.copy()
        taper_mock.return_value = np.zeros((3, 25))
        cb.taper(25)
        taper_mock.assert_called_once_with(mock.ANY, cb.stats, 25)
        np.testing.assert_array_equal(cb.data, np.zeros((3, 25)))
        np.testing.assert_array_equal(
            taper_mock.call_args[0][0], self.cb.data)
        self.assertIn('tapered: width=25s', cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_taper_center')
    def test_taper_center(self, taper_mock):
        cb = self.cb.copy()
        taper_mock.return_value = np.zeros((3, 25))
        cb.taper_center(25, 1)
        taper_mock.assert_called_once_with(
            mock.ANY, cb.stats, 25, slope_frac=1)
        np.testing.assert_array_equal(cb.data, np.zeros((3, 25)))
        np.testing.assert_array_equal(
            taper_mock.call_args[0][0], self.cb.data)
        self.assertIn(
            'tapered-centre: width=25s, slope_frac=1',
            cb.stats.processing_bulk)

    @mock.patch('seismic.correlate.stream.pcp.corr_mat_trim')
    def test_trim(self, trim_mock):
        cb = self.cb.copy()
        stats = deepcopy(cb.stats)
        stats.sampling_rate = 0.5
        trim_mock.return_value = (np.zeros((3, 25)), stats)
        cb.trim('start', 'stop')
        trim_mock.assert_called_once_with(
            mock.ANY, self.cb.stats, 'start', 'stop')
        np.testing.assert_array_equal(cb.data, np.zeros((3, 25)))
        np.testing.assert_array_equal(trim_mock.call_args[0][0], self.cb.data)
        self.assertEqual(cb.stats, stats)

    @mock.patch('seismic.correlate.stream.WFC')
    @mock.patch('seismic.correlate.stream.wfc_multi_reftr')
    def test_wfc(self, wfc_func_mock, wfc_mock):
        wfc_func_mock.return_value = {'test': 0}
        self.cb.wfc('myreftr', 'lala', 'both', 0, 15, 1, 2, False)
        wfc_func_mock.assert_called_once_with(
            mock.ANY, 'myreftr', 'lala', 'both', False)
        statsexp = deepcopy(self.cb.stats)
        statsexp['tw_start'] = 0
        statsexp['tw_len'] = 15
        statsexp['freq_min'] = 1
        statsexp['freq_max'] = 2
        wfc_mock.assert_called_once_with({'test': 0}, statsexp)

    def test_find_index_partial(self):
        start = self.cb.stats.corr_start[0] + 1
        end = self.cb.stats.corr_end[-1] - 1
        out = self.cb._find_slice_index(start, end, True)
        self.assertEqual(len(np.nonzero(out)[0]), len(self.cb.data))

    def test_find_index_not_partial(self):
        start = self.cb.stats.corr_start[0] + 1
        end = self.cb.stats.corr_end[-1] - 1
        out = self.cb._find_slice_index(start, end, False)
        self.assertEqual(len(np.nonzero(out)[0]), len(self.cb.data)-2)

    def test_find_index_before_and_after(self):
        start = self.cb.stats.corr_start[0] - 1
        end = self.cb.stats.corr_end[-1] + 1
        out = self.cb._find_slice_index(start, end, True)
        self.assertEqual(len(np.nonzero(out)[0]), len(self.cb.data))

    def test_find_index_end_before_start(self):
        end = self.cb.stats.corr_start[0]
        start = self.cb.stats.corr_end[-1]
        with self.assertRaises(ValueError):
            self.cb._find_slice_index(start, end, True)

    def test_find_index_empty(self):
        start = self.cb.stats.corr_start[0] - 15
        end = self.cb.stats.corr_start[0] - 1
        with warnings.catch_warnings(record=True) as w:
            out = self.cb._find_slice_index(start, end, True)
            self.assertEqual(len(w), 1)
        self.assertEqual(len(np.nonzero(out)[0]), 0)


class TestReadCorrBulk(unittest.TestCase):
    @mock.patch('seismic.correlate.stream.np.load')
    @mock.patch('seismic.correlate.stream.m3ut.load_header_from_np_array')
    @mock.patch('seismic.correlate.stream.CorrBulk')
    def test_read(self, cb_mock, load_header_mock, np_load_mock):
        np_load_mock.return_value = {'something': 0, 'data': 'ishere'}
        load_header_mock.return_value = {'stats': 'arehere'}
        stream.read_corr_bulk('/path/to/file')
        np.load.assert_called_once_with('/path/to/file')
        load_header_mock.assert_called_once_with(
            {'something': 0, 'data': 'ishere'})
        cb_mock.assert_called_once_with('ishere', stats={'stats': 'arehere'})


class TestCorrStats(unittest.TestCase):
    def setUp(self):
        self.cst = stream.CorrStats()

    def test_defaults(self):
        defaults = {
            'sampling_rate': 1.0,
            'delta': 1.0,
            'starttime': UTCDateTime(0),
            'endtime': UTCDateTime(0),
            'corr_start': UTCDateTime(0),
            'corr_end': UTCDateTime(0),
            'start_lag': 0,
            'end_lag': 0,
            'npts': 0,
            'calib': 1.0,
            'network': '',
            'station': '',
            'location': '',
            'channel': '',
        }
        for k in defaults:
            self.assertEqual(defaults[k], self.cst[k])

    def test_native_types(self):
        # network, station, and channel are strings and do allow nothing but
        # strings (convertibles types will be converted to str)
        keys = ['network', 'station', 'channel']
        for k in keys:
            with warnings.catch_warnings(record=True) as w:
                self.cst[k] = [1, 2, 3]
                self.assertEqual(len(w), 1)
                self.assertIsInstance(self.cst[k], str)
        # In the original obspy version this is the case for location, too.
        # Should not be here though
        self.cst['location'] = [1, 2, 3]
        self.assertIsInstance(self.cst['location'], list)

    def test_set_comp(self):
        self.cst['channel'] = 'HHE-HHZ'
        self.cst['component'] = 'R-R'
        self.assertEqual(self.cst.channel, 'HHR-HHR')

    def test_read_only(self):
        readonly = ['endtime', 'end_lag', 'starttime']
        for k in readonly:
            with self.assertRaises(AttributeError):
                self.cst[k] = 'test'

    def test_link_corr_start_starttime(self):
        # corr_start and starttime are just aliases
        st = np.random.random(10)*3600
        for t in st:
            self.cst['corr_start'] = UTCDateTime(t)
            self.assertEqual(self.cst['corr_start'], self.cst['starttime'])

    def test_link_corr_end_endtime(self):
        # corr_start and starttime are just aliases
        st = np.random.random(10)*3600
        for t in st:
            self.cst['corr_end'] = UTCDateTime(t)
            self.assertEqual(self.cst['corr_end'], self.cst['endtime'])

    def test_link_sampling(self):
        # sampling is linked (the parameters npts, sampling_rate, delta,
        # start_lag, end_lag)
        self.cst.npts = np.random.randint(2, 250)
        self.assertEqual(
            self.cst.end_lag, self.cst.start_lag+self.cst.delta*float(
                self.cst.npts-1))
        self.cst.sampling_rate = np.random.randint(2, 10)
        self.assertAlmostEqual(self.cst.delta, 1/self.cst.sampling_rate)
        self.assertEqual(
            self.cst.end_lag, self.cst.start_lag+self.cst.delta*float(
                self.cst.npts-1))
        self.cst.start_lag += np.random.randint(1, 500)
        self.assertEqual(
            self.cst.end_lag, self.cst.start_lag+self.cst.delta*float(
                self.cst.npts-1))

    def get_component(self):
        cst = deepcopy(self.cst)
        # Set channel
        cst['channel'] = 'HHE-HHZ'
        self.assertEqual(cst['component'], 'E-Z')


class TestCombineStats(unittest.TestCase):
    def setUp(self):
        self.inv = read_inventory()
        self.st = read()

    def test_result(self):
        # We test this by using two different components
        st0 = self.st[0].stats
        st1 = self.st[1].stats
        lag = np.random.randint(80, 120)
        stc = stream.combine_stats(st0, st1, -lag, inv=self.inv)
        self.assertEqual(stc.dist, 0)
        keys = ['network', 'station', 'channel', 'location']
        for k in keys:
            self.assertEqual(stc[k], '%s-%s' % (st0[k], st1[k]))
        self.assertEqual(stc.corr_start, st1.starttime)
        self.assertEqual(stc.corr_end, st1.endtime)
        self.assertEqual(stc.start_lag, -lag)

        self.assertEqual(stc.end_lag, -lag+(st0.npts-1)*st0.delta)

    def test_wrong_input(self):
        with self.assertRaises(TypeError):
            stream.combine_stats('bla', 4, 6, 3)

    def test_wrong_input2(self):
        with self.assertRaises(TypeError):
            stream.combine_stats(self.st[0].stats, 4, 6, 3)

    @mock.patch('seismic.correlate.stream.m3ut.trace_calc_az_baz_dist')
    def test_add_coords(self, az_dist_mock):
        # We test this by using two different components
        az_dist_mock.return_value = (20, 160, 3000)
        st0 = self.st[0].stats
        st1 = self.st[1].stats
        st0['stla'] = 0
        st0['stlo'] = 0
        st0['stel'] = 0
        st1['stla'] = 0
        st1['stlo'] = 0
        st1['stel'] = 0
        lag = np.random.randint(80, 120)
        stc = stream.combine_stats(st0, st1, -lag, inv=self.inv)
        keys = ['stla', 'stel', 'stlo', 'evla', 'evlo', 'evel']
        for k in keys:
            self.assertEqual(stc[k], 0)
        self.assertEqual(stc['az'], 20)
        self.assertEqual(stc['baz'], 160)
        self.assertEqual(stc['dist'], 3)
        az_dist_mock.assert_called_once_with(st0, st1)

    @mock.patch('seismic.correlate.stream.m3ut.trace_calc_az_baz_dist')
    def test_add_coords2(self, az_dist_mock):
        # We test this by using two different components
        az_dist_mock.return_value = (20, 160, 3000)
        st0 = self.st[0].stats
        st1 = self.st[1].stats
        st0['sac'] = {}
        st1['sac'] = {}
        st0['sac']['stla'] = 0
        st0['sac']['stlo'] = 0
        st0['sac']['stel'] = 0
        st1['sac']['stla'] = 0
        st1['sac']['stlo'] = 0
        st1['sac']['stel'] = 0
        lag = np.random.randint(80, 120)
        stc = stream.combine_stats(st0, st1, -lag, inv=self.inv)
        keys = ['stla', 'stel', 'stlo', 'evla', 'evlo', 'evel']
        for k in keys:
            self.assertEqual(stc[k], 0)
        self.assertEqual(stc['az'], 20)
        self.assertEqual(stc['baz'], 160)
        self.assertEqual(stc['dist'], 3)
        az_dist_mock.assert_called_once_with(st0, st1)


class TestCompareTr(unittest.TestCase):
    def setUp(self):
        self.st = read()

    def test_identical(self):
        self.assertTrue(stream.compare_tr_id(self.st[0], self.st[0]))

    def test_regard_loc(self):
        tr1 = self.st[0].copy()
        tr1.stats['location'] = '99'
        self.assertFalse(stream.compare_tr_id(self.st[0], tr1))
        self.assertTrue(
            stream.compare_tr_id(self.st[0], tr1, regard_loc=False))

    def test_different(self):
        self.assertFalse(
            stream.compare_tr_id(self.st[0], self.st[1], regard_loc=False))


class TestStackSt(unittest.TestCase):
    def setUp(self):
        self.st = Stream(traces=[read()[0]])
        tr = self.st[0]
        tr.data = np.ones_like(tr.data)
        tr.stats['corr_start'] = tr.stats.starttime
        tr.stats['corr_end'] = tr.stats.endtime
        self.corr_len = tr.stats['corr_end'] - tr.stats['corr_start']
        delta = tr.stats.endtime - tr.stats.starttime
        for ii in range(np.random.randint(3, 12)):
            ntr = tr.copy()
            ntr.data = tr.data*(ii+2)
            ntr.stats['corr_start'] += (ii+1) * delta
            ntr.stats['corr_end'] += (ii+1) * delta
            # Shorten correlation with every iteration
            ntr.stats['corr_end'] -= self.corr_len*(ii+1)*.05
            self.st.append(ntr)

    def test_result_mean(self):
        stack = stream.stack_st(self.st, 'mean', norm=False)
        self.assertEqual(len(stack), self.st[0].stats.npts)
        self.assertEqual(self.st[0].stats.corr_start, stack.stats.corr_start)
        self.assertEqual(self.st[-1].stats.corr_end, stack.stats.corr_end)
        exp_res = np.average(np.arange(1, self.st.count()+1))*np.ones_like(
            stack.data)
        self.assertTrue(np.all(stack.data == exp_res))

    def test_weighting(self):
        stack = stream.stack_st(self.st, 'by_length', norm=False)
        self.assertEqual(self.st[0].stats.corr_start, stack.stats.corr_start)
        self.assertEqual(self.st[-1].stats.corr_end, stack.stats.corr_end)
        lengths = self.corr_len*np.linspace(
            1, 1-0.05*(self.st.count()-1), self.st.count())
        exp_res = np.sum(lengths*np.arange(1, self.st.count()+1))/lengths.sum()
        self.assertTrue(np.allclose(stack.data, exp_res))
        self.assertEqual(self.st[0].stats.corr_start, stack.stats.corr_start)
        self.assertEqual(self.st[-1].stats.corr_end, stack.stats.corr_end)


class TestStackByGroup(unittest.TestCase):
    def setUp(self):
        st = read()
        self.st = stream.CorrStream()
        for tr in st:
            delta = tr.stats.endtime - tr.stats.starttime
            tr.data = np.ones_like(tr.data)
            tr.stats['corr_start'] = tr.stats.starttime
            tr.stats['corr_end'] = tr.stats.endtime
            self.corr_len = tr.stats['corr_end'] - tr.stats['corr_start']
            tr = stream.CorrTrace(tr.data, _header=tr.stats)
            for ii in range(np.random.randint(3, 12)):
                ntr = tr.copy()
                ntr.data = tr.data*(ii+2)
                ntr.stats['corr_start'] += delta*(ii+1)
                ntr.stats['corr_end'] += delta*(ii+1)
                ntr.stats.location = str(ii+5)
                self.st.append(ntr)

    def test_grouping(self):
        stack = stream.stack_st_by_group(self.st, False, 'mean')
        self.assertEqual(stack.count(), 3)

    def test_grouping_regard_loc(self):
        stack = stream.stack_st_by_group(self.st, True, 'mean')
        self.assertEqual(stack.count(), self.st.count())


class TestAlphabeticalCorrelation(unittest.TestCase):
    def setUp(self):
        st = read()
        self.st0 = st[0].stats
        self.st0.station = 'AA'
        self.st1 = st[1].stats
        self.st1.station = 'ZZ'
        self.data = np.linspace(0, 50, 51)

    def test_no_modification(self):
        header, data_out = stream.alphabetical_correlation(
            self.st0, self.st1, -10, 11, self.data.copy(), None)
        self.assertEqual(header.station, 'AA-ZZ')
        self.assertTrue(np.all(data_out == self.data))
        self.assertEqual(header.start_lag, -10)

    def test_modified(self):
        header, data_out = stream.alphabetical_correlation(
            self.st1, self.st0, -10, 11, self.data.copy(), None)
        self.assertEqual(header.station, 'AA-ZZ')
        self.assertTrue(np.all(data_out == np.flip(self.data)))
        self.assertEqual(header.start_lag, -11)

    def test_priority(self):
        self.st0.network = 'ZZ'
        header, _ = stream.alphabetical_correlation(
            self.st1, self.st0, -10, 11, self.data.copy(), None)
        self.assertEqual(header.station, 'ZZ-AA')
        self.assertEqual(header.start_lag, -10)


class TestCorrTrace(unittest.TestCase):
    # not a whole lot to test here
    def setUp(self):
        self.st = read()

    def test_npts(self):
        npts = np.random.randint(25, 510)
        data = np.empty(npts)
        ctr = stream.CorrTrace(data)
        self.assertEqual(ctr.stats.npts, npts)

    def test_provide_header(self):
        ctr = stream.CorrTrace(self.st[0].data, _header=self.st[0].stats)
        self.assertEqual(ctr.stats, stream.CorrStats(self.st[0].stats))

    def test_combine_headers(self):
        ctr = stream.CorrTrace(
            np.empty(25), header1=self.st[0].stats, header2=self.st[1].stats,
            start_lag=-10, end_lag=10)
        exp_res, _ = stream.alphabetical_correlation(
            self.st[0].stats, self.st[1].stats, -10, 10, np.empty(25), None)
        exp_res.npts = 25
        self.assertEqual(ctr.stats, exp_res)

    def test_no_header(self):
        ctr = stream.CorrTrace(
            np.empty(25))
        exp_res = stream.CorrStats()
        exp_res['npts'] = 25
        self.assertEqual(ctr.stats, exp_res)

    def test_times(self):
        ctr = stream.CorrTrace(
            np.empty(25), header1=self.st[0].stats, header2=self.st[1].stats,
            start_lag=-10, end_lag=10)
        exp = np.arange(
            ctr.stats.start_lag, ctr.stats.end_lag + ctr.stats.delta,
            ctr.stats.delta)
        np.testing.assert_array_almost_equal(exp, ctr.times())


class TestCorrStream(unittest.TestCase):
    def setUp(self):
        st = read()
        for tr in st:
            # Else this will need too much time
            tr.stats.sampling_rate = 1
        self.st = stream.CorrStream()
        for tr in st:
            delta = tr.stats.endtime - tr.stats.starttime
            tr.data = np.ones_like(tr.data)
            tr.stats['corr_start'] = tr.stats.starttime
            tr.stats['corr_end'] = tr.stats.endtime
            self.corr_len = delta
            tr = stream.CorrTrace(tr.data, _header=tr.stats)
            for ii in range(60):
                ntr = tr.copy()
                ntr.data = np.empty(tr.data.shape)
                ntr.stats['corr_start'] += delta*(ii+1)
                ntr.stats['corr_end'] += delta*(ii+1)
                self.st.append(ntr)
        self.len_tot = self.st[-1].stats.corr_end - self.st[0].stats.corr_start

    def test_wrong_input(self):
        with self.assertRaises(TypeError):
            stream.CorrStream('bla')
        with self.assertRaises(TypeError):
            stream.CorrStream([4, type])

    def test_slide_short_window_error(self):
        with self.assertRaises(ValueError):
            for _ in self.st.slide(1, 1):
                continue

    def test_slide_max_len(self):
        for st in self.st.slide(self.corr_len, self.corr_len, True):
            # Just check that no extra correlations are yielded
            self.assertEqual(st.count(), 3)
            st.sort(keys=['corr_start'])
            length = st[-1].stats.corr_end - st[0].stats.corr_start
            self.assertEqual(length, self.corr_len)

    def test_slide_full_data(self):
        # Make sure that no data is lost
        outst = stream.CorrStream()
        for st in self.st.slide(3600, 3600, True):
            outst.extend(st)
        start = min(tr.stats.corr_start for tr in outst)
        end = max(tr.stats.corr_end for tr in outst)
        self.assertEqual(start, self.st[0].stats.corr_start)
        self.assertEqual(end, self.st[-1].stats.corr_end)

    def test_slide_wrong_step_value(self):
        step = np.random.randint(-35, 0)
        with self.assertRaises(ValueError):
            for _ in self.st.slide(self.corr_len, step):
                continue

    def test_select_corr_time_no_partial(self):
        start = self.st[0].stats.corr_start
        end = self.st[0].stats.corr_end + 1
        st_out = self.st.select_corr_time(start, end, False)
        endmax = max(tr.stats.corr_end for tr in st_out)
        self.assertEqual(endmax, self.st[0].stats.corr_end)

    def test_select_corr_time_no_partial2(self):
        start = self.st[0].stats.corr_start + 1
        end = self.st[0].stats.corr_end + 1
        st_out = self.st.select_corr_time(start, end, False)
        self.assertFalse(len(st_out))

    def test_select_corr_time_w_partial(self):
        start = self.st[0].stats.corr_start + 1
        end = self.st[0].stats.corr_end + 1
        st_out = self.st.select_corr_time(start, end, True)
        startmin = min(tr.stats.corr_start for tr in st_out)
        endmax = max(tr.stats.corr_end for tr in st_out)
        self.assertEqual(endmax, self.st[0].stats.corr_end)
        self.assertEqual(startmin, self.st[0].stats.corr_start)

    def test_select_corr_time_correct_length_False(self):
        start = self.st[0].stats.corr_start
        end = self.st[0].stats.corr_start + 3*self.corr_len
        st_out = self.st.select_corr_time(start, end, False)
        st_out.sort(keys=['corr_start'])
        length = st_out[-1].stats.corr_end - st_out[0].stats.corr_start
        self.assertEqual(length, 3*self.corr_len)

    def test_select_corr_time_correct_length_True(self):
        start = self.st[0].stats.corr_start
        end = self.st[0].stats.corr_start + 3*self.corr_len
        st_out = self.st.select_corr_time(start, end, True)
        st_out.sort(keys=['corr_start'])
        length = st_out[-1].stats.corr_end - st_out[0].stats.corr_start
        self.assertEqual(length, 3*self.corr_len)

    def test_select_corr_time_correct_length_1_True(self):
        # This has been an issue
        # Because this is a special case
        start = self.st[0].stats.corr_start
        end = self.st[0].stats.corr_start + self.corr_len
        st_out = self.st.select_corr_time(start, end, True)
        st_out.sort(keys=['corr_start'])
        length = st_out[-1].stats.corr_end - st_out[0].stats.corr_start
        self.assertEqual(length, self.corr_len)

    def test_select_corr_time_correct_length_1_False(self):
        # This has been an issue
        # Because this is a special case
        start = self.st[0].stats.corr_start
        end = self.st[0].stats.corr_start + self.corr_len
        st_out = self.st.select_corr_time(start, end, False)
        st_out.sort(keys=['corr_start'])
        length = st_out[-1].stats.corr_end - st_out[0].stats.corr_start
        self.assertEqual(length, self.corr_len)

    def test_stack_len_0(self):
        out = self.st.stack(regard_location=False)
        self.assertEqual(out.count(), 3)

    def test_stack_len_daily(self):
        out = self.st.stack(stack_len='daily', regard_location=False)
        # Note that this does not mean that each correlation encompasses 24h
        # of data due to gaps at the beginning or end or overlaps over the
        # different days.
        self.assertEqual(3*np.ceil(self.len_tot/(24*3600)), out.count())

    def test_stack_len_rand(self):
        stacklen = self.corr_len*np.random.randint(2, 9)
        out = self.st.stack(stack_len=stacklen, regard_location=False)
        # for tr in out[]:
        #     # The last three could be shorter
        tr = out[4]
        self.assertEqual(stacklen, tr.stats.corr_end-tr.stats.corr_start)

    @mock.patch('seismic.correlate.stream.convert_statlist_to_bulk_stats')
    def test_create_corrbulk(self, cstbs_mock):
        st = self.st.copy()
        qustart = st[0].stats.corr_start + 1
        cstbs_mock.return_value = st[0].stats
        with mock.patch.object(st, 'select') as sct_mock:
            select_return = mock.MagicMock(name='select_mock')
            select_return.select_corr_time.return_value = st
            sct_mock.return_value = select_return
            cb = st.create_corr_bulk(times=(qustart, st[0].stats.corr_end))
            sct_mock.assert_any_call(
                None, None, None, None)
        select_return.select_corr_time.assert_called_once_with(
            qustart, st[0].stats.corr_end
        )
        for ii, tr in enumerate(self.st):
            np.testing.assert_array_equal(tr.data, cb.data[ii])
        # Check in-place
        for tr in st:
            self.assertFalse(hasattr(tr, 'data'))

    @mock.patch('seismic.correlate.stream.convert_statlist_to_bulk_stats')
    def test_create_corrbulk_differing_sampling(self, cstbs_mock):
        st = self.st.copy()
        cstbs_mock.return_value = st[0].stats
        st[-1].stats.sampling_rate /= 2
        with warnings.catch_warnings(record=True) as w:
            cb = st.create_corr_bulk()
            self.assertEqual(len(w), 1)
        self.assertEqual(cb.data.shape[0], self.st.count()-1)
        for ii, tr in enumerate(self.st[:-1]):
            np.testing.assert_array_equal(tr.data, cb.data[ii])

    def test_to_matrix(self):
        st = self.st.copy()
        with mock.patch.object(st, 'select') as sct_mock:
            select_return = mock.MagicMock(name='select_mock')
            select_return.select_corr_time.return_value = st
            sct_mock.return_value = select_return
            A, stats = st._to_matrix(
                'net', 'stat', 'cha', 'loc', times=(
                    st[0].stats.starttime, st[0].stats.endtime))
            sct_mock.assert_called_once_with('net', 'stat', 'loc', 'cha')
            select_return.select_corr_time.assert_called_once_with(
                st[0].stats.starttime, st[0].stats.endtime)
        statl = []
        for ii, tr in enumerate(st):
            np.testing.assert_array_equal(A[ii], tr.data)
            statl.append(tr.stats)
        self.assertListEqual(statl, stats)


class TestConvertStatlistToBulkStats(unittest.TestCase):
    def setUp(self):
        self.stats = read()[0].stats
        self.stats['corr_start'] = UTCDateTime(0)
        self.stats['corr_end'] = UTCDateTime(3600)
        self.stats['dist'] = 1000
        self.stats['az'] = 30
        self.stats['baz'] = 330
        self.stats['start_lag'] = -100
        self.stats['end_lag'] = 100
        self.stats['stla'] = 55
        self.stats['evla'] = 54.5
        self.stats['stlo'] = 0
        self.stats['evlo'] = 0
        self.stats['stel'] = 0
        self.stats['evel'] = 0
        # Note that starttime and endtime are just copies of corr_start and
        # corr_end
        self.mutables = ['corr_start', 'corr_end', 'starttime', 'endtime']
        self.immutables = [
            'npts', 'sampling_rate', 'network', 'station', 'channel',
            'start_lag', 'stla', 'stlo', 'stel', 'evla', 'evlo',
            'evel', 'dist', 'az', 'baz', 'location']
        self.stats = stream.CorrStats(self.stats)

    def test_result(self):
        stats1 = self.stats.copy()
        stats1['corr_start'] += 3600
        stats1['corr_end'] += 3600
        stcomb = stream.convert_statlist_to_bulk_stats([self.stats, stats1])
        for key in stcomb:
            if key in self.mutables:
                self.assertEqual(len(stcomb[key]), 2)
                self.assertEqual(stcomb[key][0], self.stats[key])
                self.assertEqual(stcomb[key][1], stats1[key])
            elif key == 'ntrcs':
                self.assertEqual(stcomb[key], 2)
            else:
                self.assertEqual(stcomb[key], self.stats[key])

    def test_differing_immutables(self):
        stats1 = self.stats.copy()
        change = np.random.randint(0, len(self.immutables)-1)
        chkey = self.immutables[change]
        stats1[chkey] = '3'
        with self.assertRaises(ValueError):
            _ = stream.convert_statlist_to_bulk_stats([self.stats, stats1])

    def test_loc_mutable(self):
        stats1 = self.stats.copy()
        stats1['location'] = 'bla'
        stcomb = stream.convert_statlist_to_bulk_stats(
            [self.stats, stats1], True)
        self.assertIsInstance(stcomb['location'], list)


if __name__ == "__main__":
    unittest.main()
