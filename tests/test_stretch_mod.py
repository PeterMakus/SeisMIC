'''
:copyright:

:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 24th June 2021 02:23:40 pm
Last Modified: Thursday, 24th June 2021 03:01:34 pm
'''

import unittest

import numpy as np

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
# class TestVelocityChangeEstimate(unittest.TestCase):
    



if __name__ == "__main__":
    unittest.main()