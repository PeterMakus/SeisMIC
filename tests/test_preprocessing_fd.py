'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    `EUROPEAN UNION PUBLIC LICENCE v. 1.2
    <https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>`_
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 20th July 2021 04:00:46 pm
Last Modified: Wednesday, 28th June 2023 02:02:47 pm
'''
import unittest
from unittest.mock import patch

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
        A = np.tile(A, (2, 1))
        out = ppfd.FDfilter(A, args, self.params)
        np.testing.assert_allclose(out.real, 0, atol=1e-4)


class TestFDSignBitNormalisation(unittest.TestCase):
    # Not much to test here
    def test_result(self):
        np.random.seed(2)
        dim = (np.random.randint(2, 6), np.random.randint(200, 766))
        A = np.random.random(dim)-.5
        B = np.fft.rfft(A)
        # to acccount for inaccuracies in the FFT and the fact
        # that the imaginary part will be ommited
        expected_result = np.fft.rfft(np.sign(np.fft.irfft(B)))
        np.testing.assert_allclose(
            expected_result, ppfd.FDsignBitNormalization(B, {}, {}))


class TestSpectralWhitening(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        A = (rng.normal(size=600) + rng.normal(size=600)*1j)
        self.A = np.reshape(A, (6, 100))

    def test_nonorm(self):
        # Again so straightforward that I wonder whether it makes sense
        # to test this
        expected = self.A/abs(self.A)
        expected[:, 0] = 0.j
        self.assertTrue(np.allclose(
            expected, ppfd.spectralWhitening(self.A, {}, {})))
        self.assertTrue(np.allclose(
            expected, ppfd.spectralWhitening(
                self.A, {"joint_norm": False}, {})))
        self.assertTrue(np.allclose(
            expected, ppfd.spectralWhitening(
                self.A, {"joint_norm": 1}, {})))

    def test_joint_norm_not_possible(self):
        with self.assertRaises(AssertionError):
            ppfd.spectralWhitening(
                np.ones((5, 5)), {'joint_norm': True}, {})

    def test_joint_norm(self):
        for v in [2, 3, True]:
            args = {"joint_norm": v}
            if v is True:
                k = 3
            else:
                k = args["joint_norm"]
            absA = np.abs(self.A)
            norm = np.repeat(np.mean(absA.reshape(-1, k, self.A.shape[1]),
                                     axis=1), k, axis=0)
            expected = np.true_divide(self.A, norm)
            expected[:, 0] = 0.j
            self.assertTrue(np.allclose(
                expected, ppfd.spectralWhitening(self.A, args, {})))

    def test_joint_norm_invalid(self):
        with self.assertRaises(ValueError):
            ppfd.spectralWhitening(
                self.A, {"joint_norm": 4}, {})
        with self.assertRaises(ValueError):
            ppfd.spectralWhitening(
                self.A, {"joint_norm": "a"}, {})

    def test_empty_array(self):
        A = np.array([])
        with self.assertRaises(IndexError):
            ppfd.spectralWhitening(
                A, {}, {})

    def test_floating_point_error(self):
        with patch('seismic.correlate.preprocessing_fd.np.true_divide') as dm:
            dm.side_effect = FloatingPointError
            ppfd.spectralWhitening(self.A, {}, {})


if __name__ == "__main__":
    unittest.main()
