"""
Micro-benchmark for the A5 fix: combination (re)calculation cost.

Times calc_cross_combis over an NSTA x NCOMP trace stream for the
scenarios that exercise the different code paths:

- ``empty``:   ex_dict empty -> every pair runs the full flip/key
               construction and the KeyError exit.
- ``full``:    ex_dict holds every pair -> every pair does the nested
               dict traversal and is rejected; is_in_xcombis never runs.
- ``full_rc``: like full, but with rcombis set. NOTE: with a fully
               populated ex_dict, is_in_xcombis is never reached either;
               to exercise the fnmatch path use ``empty_rc``.
- ``empty_rc``: ex_dict empty + rcombis -> is_in_xcombis per pair.

Repeat-window behavior is the relevant one for A5's caches, so the
scenario is run once to warm the module-level caches and then timed
(cold = first window, warm = every subsequent window).

Untracked helper for the per-fix performance branches (perf/a5-...).
Not part of the package; do not commit without a decision.
"""
import argparse
import time

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from seismic.correlate.correlate import calc_cross_combis

NSTA = 100
NCOMP = 3


def make_stream():
    st = Stream()
    for i in range(NSTA * NCOMP):
        tr = Trace(np.arange(10, dtype=np.float32))
        tr.stats.network = "XX"
        tr.stats.station = "S%03d" % (i // NCOMP)
        tr.stats.channel = "BH" + "ZNE"[i % NCOMP]
        tr.stats.starttime = UTCDateTime(2026, 1, 1)
        tr.stats.sampling_rate = 50.0
        st.traces.append(tr)
    return st.sort()


def make_ex_dict(populated):
    if not populated:
        return {}
    d = {}
    t0 = UTCDateTime(2026, 1, 1).format_fissures()
    for i in range(NSTA):
        s0 = {}
        for j in range(NSTA):
            s0[f"XX.S{j:03d}"] = {"--": {"BHZ-BHZ": [t0]}}
        d[f"XX.S{i:03d}"] = s0
    return d


RCOMBIS = ["XX.S0*-XX.S1*", "XX.S2*-XX.S3*"]

SCENARIOS = {
    "empty": (False, None),
    "full": (True, None),
    "empty_rc": (False, RCOMBIS),
    "full_rc": (True, RCOMBIS),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    st = make_stream()
    for name, (populated, rcombis) in SCENARIOS.items():
        ex = make_ex_dict(populated)
        n = len(calc_cross_combis(st, ex, "betweenStations", rcombis))
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            calc_cross_combis(st, ex, "betweenStations", rcombis)
            times.append(time.perf_counter() - t0)
        print("%-9s pairs=%5d  warm: %s ms" % (
            name, n, ["%.0f" % (t * 1e3) for t in times]))


if __name__ == "__main__":
    main()
