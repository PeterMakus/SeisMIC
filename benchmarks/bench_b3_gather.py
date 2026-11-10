"""
Micro-benchmark for the B3 fix: window exchange cost.

Compares the old exchange (allgather of pickled obspy Streams) with the
packed exchange (allgather of (headers, flat data, masks) tuples +
reconstruction, buffers freed right after unpacking) at the current rank
count. Every rank holds ntrc/psize traces of one window; every 5th trace
carries an edge mask, as produced by generate_corr_inc's trim(pad=True).

Run with:  mpirun -np <ranks> python benchmarks/bench_b3_gather.py

Prints one JSON line per rank. Untracked helper for perf/b3-...;
not part of the package; do not commit without a decision.
"""
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from mpi4py import MPI
from obspy import Stream, Trace, UTCDateTime

from seismic.correlate.correlate import _pack_window, _unpack_window

NTRC = 300
NPTS = 90001
FS = 50.0
REPS = 3


def rss_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux KiB
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def make_window(rank, psize):
    n = NTRC // psize
    st = Stream()
    for i in range(n):
        tr_id = rank * n + i
        d = np.arange(NPTS, dtype=np.float32) + tr_id
        if tr_id % 5 == 0:
            d = np.ma.array(d, mask=np.zeros(NPTS, dtype=bool))
            d[:100] = np.ma.masked
        tr = Trace(d)
        tr.stats.network = "XX"
        tr.stats.station = "S%03d" % (tr_id // 3)
        tr.stats.channel = "BH" + "ZNE"[tr_id % 3]
        tr.stats.location = ""
        tr.stats.sampling_rate = FS
        tr.stats.starttime = UTCDateTime(2026, 1, 1)
        st.traces.append(tr)
    return st


def old_exchange(comm, win):
    winl = comm.allgather(win)
    win = Stream()
    for winp in winl:
        win.extend(winp)
    win = win.sort()
    return win


def new_exchange(comm, win):
    winl = comm.allgather(_pack_window(win))
    win = Stream()
    for winp in winl:
        win.extend(_unpack_window(winp))
    del winl
    win = win.sort()
    return win


def check(out):
    """Spot-check trace order, values, masks and dtypes."""
    # the windows are sorted by (station, channel[, ...]); channels sort
    # alphabetically (BHE < BHN < BHZ), not ZNE
    order = sorted(range(NTRC), key=lambda t: (
        "S%03d" % (t // 3), "BH" + "ZNE"[t % 3]))
    ok = len(out) == NTRC
    for i, tr_id in enumerate(order):
        tr = out[i]
        want = np.arange(NPTS, dtype=np.float32) + tr_id
        masked = tr_id % 5 == 0
        if not (
            tr.id == f"XX.S{tr_id // 3:03d}..BH{'ZNE'[tr_id % 3]}"
            and isinstance(tr.data, np.ma.MaskedArray if masked else np.ndarray)
            and tr.data.dtype == np.float32
            and np.ma.allequal(tr.data, np.ma.array(want, mask=(
                np.arange(NPTS) < 100 if masked else np.ma.nomask)))
        ):
            ok = False
            break
    return ok


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    psize = comm.Get_size()

    results = {}
    for name, ex in (("stream", old_exchange), ("packed", new_exchange)):
        win = make_window(rank, psize)
        rss0 = rss_mb()
        # correctness once: both paths must give the same stream
        out = ex(comm, win)
        ok = check(out)
        ok = comm.allreduce(ok, op=MPI.LAND)
        if not ok:
            # all ranks exit together, or the others would hang in the
            # next collective
            if rank == 0:
                print("FAIL: exchange produced wrong stream", flush=True)
            sys.exit(1)
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ex(comm, win)
            ts.append(time.perf_counter() - t0)
        rss1 = rss_mb()
        results[name] = {
            "times_ms": [round(t * 1e3, 1) for t in ts],
            "peak_rss_mb": round(rss1, 1),
            "rss_before_mb": round(rss0, 1),
        }
        del win, out

    results["paths_agree"] = True

    print(json.dumps({
        "label": "B3-gather", "rank": rank, "psize": psize,
        "ntrc": NTRC, "npts": NPTS, **results,
    }), flush=True)


if __name__ == "__main__":
    main()
