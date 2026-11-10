"""
Micro-benchmark for the A4 fix: CorrTrace header combination cost.

Measures how long CorrTrace.__init__ (i.e. combine_stats) takes per pair
with and without obspy Response objects attached to the trace stats --
the per-pair ``stats1[key] == stats2[key]`` loop deep-compares the
response tree, which is discarded by combine_stats anyway.

Untracked helper for the per-fix performance branches (perf/a4-...).
Not part of the package; do not commit without a decision.
"""
import copy
import time

import numpy as np
from obspy import Stream, Trace, UTCDateTime, read_inventory

from seismic.correlate.stream import CorrTrace

NSTA = 20
NCOMP = 3
NPAIRS = NSTA * NCOMP * (NSTA * NCOMP - 1) // 2


def make_stream(with_response, resp):
    ntrc = NSTA * NCOMP
    st = Stream()
    for i in range(ntrc):
        net = "XX"
        sta = "S%03d" % (i // NCOMP)
        cha = "BH" + "ZNE"[i % NCOMP]
        tr = Trace(np.arange(10, dtype=np.float32))
        tr.stats.network = net
        tr.stats.station = sta
        tr.stats.channel = cha
        tr.stats.starttime = UTCDateTime(2026, 1, 1)
        tr.stats.sampling_rate = 50.0
        # realistic geographic coordinates in the stats (top level),
        # as SeisMIC's metadata insertion provides them
        slat, slon = 60.0 + 0.01 * (i // NCOMP), 10.0 + 0.01 * (i // NCOMP)
        tr.stats.stla, tr.stats.stlo, tr.stats.stel = slat, slon, 100.0
        if with_response:
            # attach a full inventory response tree to each trace's stats,
            # as obspy does when reading with full metadata
            tr.stats.response = copy.deepcopy(resp)
        st.traces.append(tr)
    return st


def run(with_response, resp):
    st = make_stream(with_response, resp)
    t0 = time.perf_counter()
    for i in range(len(st)):
        for j in range(i + 1, len(st)):
            CorrTrace(
                np.arange(101, dtype=np.float32),
                header1=st[i].stats, header2=st[j].stats,
                start_lag=-50.0, end_lag=50.0)
    dt = time.perf_counter() - t0
    return dt


if __name__ == "__main__":
    inv = read_inventory()
    resp = inv[0].stations[0].channels[0].response
    for label, wr in (("with response", True), ("no response", False)):
        times = [run(wr, resp) for _ in range(3)]
        best = min(times)
        print(
            "%-14s %d pairs: %.3f s (%.3f ms/pair), reps %s"
            % (label, NPAIRS, best, best / NPAIRS * 1e3,
               [round(t, 3) for t in times]))
