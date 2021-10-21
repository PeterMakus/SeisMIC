'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 04:00:46 pm
Last Modified: Thursday, 21st October 2021 02:57:39 pm
'''
import unittest

import numpy as np

from seismic.correlate import preprocessing_fd as ppfd


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
            expected, ppfd.spectralWhitening(self.A, {}, {})))

    def test_joint_norm_not_possible(self):
        with self.assertRaises(AssertionError):
            ppfd.spectralWhitening(
                np.ones((5, 5)), {'joint_norm': True}, {})

    def test_empty_array(self):
        A = np.array([])
        with self.assertRaises(IndexError):
            ppfd.spectralWhitening(
                A, {}, {})


if __name__ == "__main__":
    unittest.main()
