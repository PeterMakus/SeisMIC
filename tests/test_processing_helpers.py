'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    `EUROPEAN UNION PUBLIC LICENCE v. 1.2
    <https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>`_
:author:
   Johanna Lehr (jlehr@gfz.de)

Created: 2025-03-20 16:11:17
Last Modified: 2025-03-20 16:11:20
'''
import unittest

import numpy as np

from seismic.utils import processing_helpers as ph


class TestGetJointNorm(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=42)
        B = rng.normal(size=600)
        self.B = B.reshape(6, 100)

    def test_nonorm(self):
        ARGS = [{}, {"joint_norm": 1}, {"joint_norm": False}]
        B = np.copy(self.B)
        expected = self.B
        for args in ARGS:
            with self.subTest(args=args):
                ph.get_joint_norm(B, args)
                self.assertTrue(np.allclose(expected, B))

    def test_jointnorm(self):
        for v in [2, 3, True]:
            args = {"joint_norm": v}
            if v is True:
                k = 3
            else:
                k = args["joint_norm"]
            B = np.copy(self.B)
            expected = np.repeat(np.mean(B.reshape(-1, k, self.B.shape[1]),
                                 axis=1), k, axis=0)
            with self.subTest(args=args):
                ph.get_joint_norm(B, args)
                self.assertTrue(np.allclose(
                    expected, B))

    def test_joint_norm_invalid(self):
        with self.assertRaises(ValueError):
            ph.get_joint_norm(
                np.copy(self.B), {"joint_norm": 4})
        with self.assertRaises(ValueError):
            ph.get_joint_norm(
                np.copy(self.B), {"joint_norm": "a"})

    def test_joint_norm_not_possible(self):
        with self.assertRaises(AssertionError):
            ph.get_joint_norm(
                np.ones((5, 5)), {'joint_norm': True})


class TestSmoothRows(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=42)
        B = rng.normal(size=600)
        self.B = B.reshape(6, 100)
        self.params = {"sampling_rate": 25}

    def test_windowLength(self):
        with self.assertRaises(ValueError):
            ph.smooth_rows(self.B, {"windowLength": 0}, self.params)

    def test_windowLength_too_large(self):
        with self.assertRaises(ValueError):
            ph.smooth_rows(self.B, {"windowLength": 100}, self.params)

    def test_smoothing(self):
        args = {"windowLength": 5/25}
        B = np.copy(self.B)
        win = np.ones(int(np.ceil(args['windowLength'] *
                                  self.params['sampling_rate']))
                      ) / np.ceil(args['windowLength'] *
                                  self.params['sampling_rate'])
        for ind in range(B.shape[0]):
            B[ind, :] = np.convolve(B[ind], win, mode='same')
            B[ind, :] = np.convolve(B[ind, ::-1], win, mode='same')[::-1]
        expected = B
        self.assertTrue(np.allclose(
            expected, ph.smooth_rows(self.B, args, self.params)))

    def test_no_smoothing(self):
        args = {"windowLength": 1/25}
        expected = np.copy(self.B)
        self.assertTrue(np.allclose(
            expected, ph.smooth_rows(self.B, args, self.params)))
