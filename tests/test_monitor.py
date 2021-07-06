'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 6th July 2021 09:18:14 am
Last Modified: Tuesday, 6th July 2021 09:41:19 am
'''

import unittest

import numpy as np
from obspy import UTCDateTime

from miic3.monitor import monitor


class TestMakeTimeList(unittest.TestCase):
    def test_result(self):
        start_date = UTCDateTime(np.random.randint(0, 157788000))
        end_date = start_date + np.random.randint(31557600, 157788000)
        win_len = np.random.randint(1800, 2*86400)
        date_inc = np.random.randint(1800, 2*86400)
        st, et = monitor.make_time_list(
            start_date, end_date, date_inc, win_len)
        self.assertEqual(len(st), len(et))
        self.assertEqual(len(st), np.ceil((end_date-start_date)/date_inc))
        self.assertTrue(np.all(et-st == win_len))

    def test_end_before_start(self):
        start_date = UTCDateTime(np.random.randint(0, 157788000))
        end_date = start_date - np.random.randint(31557600, 157788000)
        win_len = np.random.randint(1800, 2*86400)
        date_inc = np.random.randint(1800, 2*86400)
        with self.assertRaises(ValueError):
            _ = monitor.make_time_list(start_date, end_date, date_inc, win_len)

    def test_neg_inc(self):
        start_date = UTCDateTime(np.random.randint(0, 157788000))
        end_date = start_date + np.random.randint(31557600, 157788000)
        win_len = np.random.randint(1800, 2*86400)
        date_inc = np.random.randint(-1800, 0)
        with self.assertRaises(ValueError):
            _ = monitor.make_time_list(start_date, end_date, date_inc, win_len)

    def test_neg_winlen(self):
        start_date = UTCDateTime(np.random.randint(0, 157788000))
        end_date = start_date + np.random.randint(31557600, 157788000)
        date_inc = np.random.randint(1800, 2*86400)
        win_len = np.random.randint(-1800, 0)
        with self.assertRaises(ValueError):
            _ = monitor.make_time_list(start_date, end_date, date_inc, win_len)


if __name__ == "__main__":
    unittest.main()
