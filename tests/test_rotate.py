'''
:copyright:
   The SeisMIC development team (jlehr@gfz.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Johanna Lehr (jlehr@gfz.de)

Created: Thursday, 23rd April 2026 04:52:36 pm
Last Modified: Thursday, 23rd April 2026 06:57:16 pm
'''

import unittest

import numpy as np

from obspy.core import Trace, Stream, Stats

from seismic.utils import rotate
from seismic.correlate.correlate import do_xcorr_in_fd
from seismic.correlate.stream import CorrStream, CorrTrace


def get_stream(X_ENZ, sname, slo, sla):
    """
    Create an obspy Stream with 3 traces for the ENZ components
    from (3, npts) array X_ENZ. Necessary stats for SeisMICs
    CorrStream are set, such as station coordinates.
    """
    st = Stream()
    for x, c in zip(X_ENZ, "ENZ"):
        stats = dict(
            network="XX",
            station=sname,
            location="",
            channel=f"HH{c}",
            sampling_rate=1,
            stla=sla,
            stlo=slo,
            stel=0,
            npts=x.size
        )
        tr = Trace(x, header=Stats(stats))
        st.append(tr)
    st.sort()
    return st


def correlate_streams(st1, st2, normalize=False):
    """
    Compute pairwise correlations of streams st1 and st2 and
    return a CorrStream with the results.
    """
    cst = CorrStream()
    for tr1 in st1:
        for tr2 in st2:
            assert tr1.stats.sampling_rate == tr2.stats.sampling_rate
            assert tr1.stats.npts == tr2.stats.npts
            x0 = np.fft.rfft(tr1.data)
            x1 = np.fft.rfft(tr2.data)
            freqs = np.fft.rfftfreq(
                tr2.data.size, 1.0 / tr2.stats.sampling_rate)
            irfft = (x0.size - 1) * 2

            xx = do_xcorr_in_fd(
                x0, x1, freqs, irfftsize=irfft, normalize=normalize)
            ctr = CorrTrace(xx, tr1.stats, tr2.stats,
                            start_lag=-tr1.stats.npts//2,
                            end_lag=tr1.stats.npts//2 + 1)
            cst.append(ctr)
    return cst


class TestRotate(unittest.TestCase):
    def setUp(self):
        """
        Create two synthetic 3-component signals with a single pulse
        (=explosion) and a known time shift between them. The signals
        come from a known direction. We assume a Euclidean geometry (
        no Earth curvature as in SeisMIC).

        For the test, we

        - compute the pairwise correlations of the two signals in the ENZ
        system and rotate the correlations to RTZ.
        - rotate the signals to RTZ and compute the pairwise correlations in
        the RTZ system.
        - compare the two results to check if the results are the same.
        """
        i0 = 7
        x0 = np.zeros(16)
        x1 = x0.copy()
        x1[i0] = 1

        # Shift second signal in time
        self.shift = -2
        x2 = np.roll(x1, self.shift)

        phi = 240  # Azimuth in degrees, wave coming from 60° at S2
        A = np.array([
            np.sin(np.radians(phi)), np.cos(np.radians(phi)), 1
            ])[:, None]
        X_ENZ1 = np.array([x1, x1, x0])*A
        X_ENZ2 = np.array([x2, x2, x0])*A*0.5

        self.sampling_rate = 1

        self.st1 = get_stream(X_ENZ1, "S1", -A[0], -A[1])
        self.st2 = get_stream(X_ENZ2, "S2", 0, 0)

        self.az = phi
        self.baz = phi-180

        self.st1_rtz = self.st1.copy().rotate(
            "NE->RT", back_azimuth=self.az-180
            ).sort()

        self.st2_rtz = self.st2.copy().rotate(
            "NE->RT", back_azimuth=self.baz
            ).sort()

    def test_rotate_corrstream_enz2rtz(self):
        normalize = False

        cst_ref = correlate_streams(self.st1_rtz, self.st2_rtz, normalize)

        cst_enz = correlate_streams(self.st1, self.st2, normalize)

        # Force az, baz
        for tr in cst_enz:
            tr.stats.az = self.az
            tr.stats.baz = self.baz
        cst_rtz = rotate.rotate_corrstream_enz2rtz(cst_enz)

        # Check if all components are 0 except for RR
        for tr in cst_rtz:
            if tr.stats.channel == "HHR-HHR":
                self.assertTrue(np.max(tr.data) == 0.5)
            else:
                self.assertTrue(np.all(np.isclose(tr.data, 0)))

        for i in range(9):
            self.assertTrue(
                np.all(np.isclose(cst_rtz[i].data, cst_ref[i].data)))

    def test_rotate_corrstream_en2rt(self):
        normalize = False

        cst_ref = correlate_streams(
            self.st1_rtz[:2], self.st2_rtz[:2], normalize)

        cst_enz = correlate_streams(
            self.st1[:2], self.st2[:2], normalize)

        # Force az, baz
        for tr in cst_enz:
            tr.stats.az = self.az
            tr.stats.baz = self.baz
        cst_rtz = rotate.rotate_corrstream_en2rt(cst_enz)

        for i in range(4):
            self.assertTrue(
                np.all(np.isclose(cst_rtz[i].data, cst_ref[i].data)))
