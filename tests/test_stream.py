'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 31st May 2021 01:50:04 pm
Last Modified: Tuesday, 1st June 2021 10:17:30 am
'''

import unittest
from unittest.mock import patch

import numpy as np
from obspy import read, read_inventory, Stream
from obspy.core import Stats

from miic3.correlate import stream


class TestCombineStats(unittest.TestCase):
    def setUp(self):
        self.inv = read_inventory()
        self.st = read()

    def test_result(self):
        # We test this by using two different components
        st0 = self.st[0].stats
        st1 = self.st[1].stats
        lag = np.random.randint(80, 120)
        stc = stream.combine_stats(st0, st1, -lag, lag, inv=self.inv)
        self.assertEqual(stc.dist, 0)
        keys = ['network', 'station', 'channel', 'location']
        for k in keys:
            self.assertEqual(stc[k], '%s-%s' % (st0[k], st1[k]))
        self.assertEqual(stc.corr_start, st1.starttime)
        self.assertEqual(stc.corr_end, st1.endtime)
        self.assertEqual(stc.start_lag, -lag)
        self.assertEqual(stc.end_lag, lag)

    def test_wrong_input(self):
        with self.assertRaises(TypeError):
            stream.combine_stats('bla', 4, 6, 3)


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
            ntr.stats['corr_start'] += delta
            ntr.stats['corr_end'] += delta
            # Shorten correlation with every iteration
            ntr.stats['corr_end'] -= self.corr_len*(ii+1)*.05
            self.st.append(ntr)

    def test_result_mean(self):
        stack = stream.stack_st(self.st, 'mean')
        self.assertEqual(len(stack), self.st[0].stats.npts)
        self.assertEqual(self.st[0].stats.corr_start, stack.stats.corr_start)
        self.assertEqual(self.st[-1].stats.corr_end, stack.stats.corr_end)
        exp_res = np.average(np.arange(1, self.st.count()+1))*np.ones_like(
            stack.data)
        self.assertTrue(np.all(stack.data == exp_res))

    def test_weighting(self):
        stack = stream.stack_st(self.st, 'by_length')
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
            self.st0, self.st1, -10, 10, self.data.copy(), None)
        self.assertEqual(header.station, 'AA-ZZ')
        self.assertTrue(np.all(data_out == self.data))

    def test_modified(self):
        header, data_out = stream.alphabetical_correlation(
            self.st1, self.st0, -10, 10, self.data.copy(), None)
        self.assertEqual(header.station, 'AA-ZZ')
        self.assertTrue(np.all(data_out == np.flip(self.data)))

    def test_priority(self):
        self.st0.network = 'ZZ'
        header, _ = stream.alphabetical_correlation(
            self.st1, self.st0, -10, 10, self.data.copy(), None)
        self.assertEqual(header.station, 'ZZ-AA')


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
        self.assertEqual(ctr.stats, self.st[0].stats)

    def test_combine_headers(self):
        ctr = stream.CorrTrace(
            np.empty(25), header1=self.st[0].stats, header2=self.st[1].stats)
        exp_res, _ = stream.alphabetical_correlation(
            self.st[0].stats, self.st[1].stats, None, None, np.empty(25), None)
        exp_res.npts = 25
        self.assertEqual(ctr.stats, exp_res)

    def test_no_header(self):
        ctr = stream.CorrTrace(
            np.empty(25))
        exp_res = Stats()
        exp_res['npts'] = 25
        self.assertEqual(ctr.stats, exp_res)


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


if __name__ == "__main__":
    unittest.main()