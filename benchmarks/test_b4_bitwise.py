"""
Bitwise A/B verification for the B4 fix (in-place preprocessing buffers).

Runs the edited TD/FD preprocessing functions on identical random inputs
and prints sha256 hashes of the outputs. Run once on the pre-B4 code and
once on the B4 code; the hashes must match.

Usage:  python benchmarks/test_b4_bitwise.py
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from seismic.correlate import preprocessing_fd as ppfd
from seismic.correlate import preprocessing_td as pptd

NTRC, NPTS, FS = 60, 90001, 50.0


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:16]


def main():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((NTRC, NPTS)).astype(np.float32)
    freqs = np.fft.rfftfreq(NPTS, 1.0 / FS)

    # TDnormalization: filter-free branch, joint_norm 3
    A1 = A.copy()
    out = pptd.TDnormalization(
        A1, {"windowLength": 5, "joint_norm": 3, "filter": None},
        {"sampling_rate": FS})
    print("TDnormalization      ", h(out), flush=True)

    # TDnormalization with a filter (TDfilter branch)
    A1b = A.copy()
    out = pptd.TDnormalization(
        A1b,
        {"windowLength": 5, "joint_norm": 3,
         "filter": {"type": "bandpass", "freqmin": 0.5, "freqmax": 2.}},
        {"sampling_rate": FS})
    print("TDnormalization+filt ", h(out), flush=True)

    # normalizeStandardDeviation, joint_norm 3
    A2 = A.copy()
    out = pptd.normalizeStandardDeviation(
        A2, {"windowLength": 5, "joint_norm": 3}, {})
    print("normalizeStdDev      ", h(out), flush=True)

    # mute with extend_gaps
    A3 = A.copy()
    out = pptd.mute(
        A3,
        {"taper_len": 1.0, "threshold": 1000.0, "std_factor": 1,
         "extend_gaps": True},
        {"sampling_rate": FS})
    print("mute                 ", h(out), flush=True)

    # FDfilter (complex64 input, float64 taper)
    B = np.fft.rfft(A, axis=1).astype(np.complex64)
    out = ppfd.FDfilter(
        B.copy(), {"flimit": [0.5, 1.0, 5.0, 10.0]}, {"freqs": freqs})
    print("FDfilter             ", h(out), flush=True)


if __name__ == "__main__":
    main()
