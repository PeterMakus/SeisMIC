"""
C1: profile one real write cycle.

Builds a realistic CorrStream (NFLES files x TRACES_PER_FILE CorrTraces
of lengthToSave samples -- one rank's share of one read_len period at
reference scale) and profiles ``_write``-equivalent writes:

- phase timings: database open (incl. correlation-option processing),
  add_correlation (dataset creation + compression), close/flush;
- written twice per file, like consecutive read_len periods (the second
  pass exercises the options-comparison path);
- cProfile breakdown for the default gzip3 case;
- total write time and output size for compression None/gzip3/gzip9.

Usage:  python benchmarks/profile_write.py [--nfiles 10] [--traces 45]
"""
import argparse
import cProfile
import io
import pstats
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from obspy import UTCDateTime

from seismic.correlate.stream import CorrStats, CorrTrace, CorrStream
from seismic.db.corr_hdf5 import CorrelationDataBase

NFILES = 10
TRACES = 45
NSAMP = 10001


def make_options(ncomb):
    return {
        "corr_args": {
            "lengthToSave": 100,
            "center_correlation": True,
            "normalize_correlation": True,
            "preProcessing": [],
        },
        "subdivision": {"corr_len": 3600.0, "corr_inc": 1800.0},
        "combinations": [(i, j) for i in range(300)
                         for j in range(i + 1, 300)][:ncomb],
        "remove_response": False,
    }


def make_header(i, j, t0):
    return CorrStats({
        "network": "XX",
        "station": "S%03d-S%03d" % (i, j),
        "location": "--",
        "channel": "BHZ-BHZ",
        "corr_start": UTCDateTime(2026, 1, 1) + t0,
        "corr_end": UTCDateTime(2026, 1, 1) + t0 + 100.0,
        "sampling_rate": 50.0,
        "start_lag": -100.0,
        "end_lag": 100.0,
        "npts": NSAMP,
    })


def make_cst(rng, t0):
    cst = CorrStream()
    for i in range(20):
        for j in range(i + 1, 20):
            cst.append(CorrTrace(
                rng.standard_normal(NSAMP).astype(np.float32),
                _header=make_header(i, j, t0)))
    return cst


def write_cycle(tmpdir, options, compression, rng, tshift=0.0):
    """One write pass over NFILES files, returns phase timings."""
    timings = {"open": 0.0, "add": 0.0, "close": 0.0}
    for f in range(NFILES):
        outf = str(Path(tmpdir) / ("XX.XX.S%03d-S%03d..BHZ-BHZ.h5"
                                   % (f, f + 1)))
        # tshift shifts the correlation windows, like consecutive
        # read_len periods do
        cst = make_cst(rng, f + tshift)
        t0 = time.perf_counter()
        cdb = CorrelationDataBase(
            outf, corr_options=options, compression=compression).__enter__()
        timings["open"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        cdb.add_correlation(cst, "subdivision")
        timings["add"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        cdb._close()
        timings["close"] += time.perf_counter() - t0
    return timings


def main():
    global NFILES, TRACES
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfiles", type=int, default=NFILES)
    ap.add_argument("--traces", type=int, default=TRACES)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    NFILES, TRACES = args.nfiles, args.traces

    ncomb = 44550  # reference scale: all pairs of 300 traces
    options = make_options(ncomb)
    rng = np.random.default_rng(3)

    for label, compression in (("None", None), ("gzip3", "gzip3"),
                               ("gzip9", "gzip9")):
        tmpdir = tempfile.mkdtemp()
        # two passes like consecutive read_len periods (the second opens
        # existing files, compares correlation options, and adds new
        # datasets)
        t = [write_cycle(tmpdir, options, compression, rng, tshift=shift)
             for shift in (0.0, 200.0)]
        size = sum(
            p.stat().st_size for p in Path(tmpdir).rglob("*.h5"))
        for k, phase in enumerate(("pass1", "pass2")):
            tt = t[k]
            print("%-6s %-5s total %6.2f s  open %5.2f  add %6.2f  "
                  "close %5.2f  (%d traces x %d samples)"
                  % (label, phase, tt["open"] + tt["add"] + tt["close"],
                     tt["open"], tt["add"], tt["close"],
                     NFILES * 190, NSAMP))
        print("%-6s output size: %.1f MB" % (label, size / 1e6))
        if label == "gzip3" and args.profile:
            shutil.rmtree(tmpdir)
            tmpdir = tempfile.mkdtemp()
            pr = cProfile.Profile()
            pr.enable()
            write_cycle(tmpdir, options, compression, rng)
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("cumulative")\
                .print_stats(25)
            print(s.getvalue())
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
