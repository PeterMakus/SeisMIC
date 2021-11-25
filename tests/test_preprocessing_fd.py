'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 04:00:46 pm
Last Modified: Thursday, 25th November 2021 10:40:14 am
'''
import unittest

import numpy as np

from seismic.correlate import preprocessing_fd as ppfd


class TestFDFilter(unittest.TestCase):
    def setUp(self):
        self.params = {
            'sampling_rate': 100}

    def test_bandpass(self):
        args = {'flimit': (42, 43, 44, 45)}
        freqs = np.fft.fftfreq(100, d=1/100)
        self.params['freqs'] = freqs
        A = np.fft.fft(np.cos(np.linspace(0, 2*np.pi, 100)))
        A = np.tile(A, (2, 1)).T
        out = ppfd.FDfilter(A, args, self.params)
        np.testing.assert_allclose(out.real, 0, atol=1e-4)


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
