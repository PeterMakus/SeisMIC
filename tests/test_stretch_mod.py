'''
:copyright:

:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 24th June 2021 02:23:40 pm
Last Modified: Tuesday, 29th June 2021 11:43:23 am
'''

import unittest

import numpy as np
from scipy.interpolate import interp1d

from miic3.monitor import stretch_mod as sm


class TestTimeWindowsCreation(unittest.TestCase):
    def test_wrong_leng(self):
        with self.assertRaises(ValueError):
            sm.time_windows_creation([0, 1, 2], [0, 2])

    def test_negative_step(self):
        win_len = np.random.randint(-100, 0)
        with self.assertRaises(ValueError):
            sm.time_windows_creation([0], win_len)

    def test_one_window(self):
        st = np.random.randint(-10, 10)
        win_len = np.random.randint(2, 100)
        out = sm.time_windows_creation([st], win_len)
        self.assertEqual(win_len, len(out[0]))
        self.assertEqual(out[0][0], st)
        

# pretty complex to test
# This is a little more like an integral test
class TestTimeStretchEstimate(unittest.TestCase):
    def setUp(self):
        self.n = 1001
        self.ref = np.sin(np.linspace(0, np.pi, self.n, endpoint=True))
        self.xref = np.linspace(0, np.pi, self.n, endpoint=True)
        # self.t = np.arange(0, 0 + 200, 1)

    def test_result(self):
        stretch = np.arange(1, 11, 1)/100  # in per cent
        # number of points for new
        nn = ((1+stretch)*self.n).round()
        corr = np.empty((len(nn), self.n))
        # inter = interp1d(self.xref, self.ref, kind='cubic')
        for ii, n in enumerate(nn):
            x = np.linspace(0, np.pi, int(n), endpoint=True)
            jj = int(round(abs(len(x)-self.n)/2))
            corr[ii, :] = np.sin(x)[jj:-jj][:self.n]
            #corr[ii, :] = inter(x[jj:-jj][:1001])
        dv = sm.time_stretch_estimate(corr, self.ref, stretch_steps=1000)
        # print(dv['value'])
        self.assertTrue(np.allclose(dv['value'], stretch))

    def test_no_stretch(self):
        corr = np.tile(self.ref, (4, 1))
        dv = sm.time_stretch_estimate(corr, self.ref, stretch_steps=101)
        print(dv)
        print(dv['value'])
        self.assertTrue(np.allclose(dv['value'], [0, 0, 0, 0]))

    



if __name__ == "__main__":
    unittest.main()