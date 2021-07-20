'''
:copyright:
:license:
   `GNU Lesser General Public License, Version 3 <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 04:00:46 pm
Last Modified: Tuesday, 20th July 2021 04:02:04 pm
'''
import unittest

import numpy as np

from miic3.correlate import preprocessing_fd as ppfd


class TestFDSignBitNormalisation(unittest.TestCase):
    # Not much to test here
    def test_result(self):
        np.random.seed(2)
        dim = (np.random.randint(200, 766), np.random.randint(2, 44))
        A = np.random.random(dim)-.5
        Aft = np.fft.rfft(A, axis=0)
        expected_result = np.fft.rfft(np.sign(A), axis=0)
        self.assertTrue(np.allclose(
            expected_result, ppfd.FDsignBitNormalization(Aft, {}, {})))


if __name__ == "__main__":
    unittest.main()
