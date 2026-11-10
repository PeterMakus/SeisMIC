"""
Benchmark the correlation kernel Correlator._pxcorr_matrix on synthetic data.

Untracked helper for the per-fix performance branches (perf/a1-..., etc.).
It is NOT part of the package and should not be committed without a decision.

For each scenario it builds a deterministic random data matrix, runs the
kernel once, and records wall time, peak RSS (process high-water mark), and
the SHA-256 of the correlation matrix C -- the hash is the bitwise regression
guard: every Phase A/B fix must reproduce the baseline hash exactly.

Usage:
    python benchmarks/bench_kernel.py [--nsta 20] [--corr-len 1800]
        [--repeats 2] [--label BASELINE]

Results are appended to benchmarks/results.jsonl (one JSON object per line)
tagged with the current git branch and commit.
"""
import argparse
import hashlib
import json
import logging
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from obspy import UTCDateTime
from mpi4py import MPI

from seismic.correlate.correlate import Correlator

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results.jsonl"


def build_correlator(options, sampling_rate):
    """Minimal Correlator shim: only what _pxcorr_matrix touches."""
    c = Correlator.__new__(Correlator)
    c.comm = MPI.COMM_WORLD
    c.rank = c.comm.Get_rank()
    c.psize = c.comm.Get_size()
    c.options = options
    c.sampling_rate = sampling_rate
    c.logger = logging.getLogger("bench_pxcorr")
    logging.basicConfig(level=logging.WARNING)
    return c


def make_options(ntrc, sampling_rate, corr_len, length_to_save, offsets):
    """offsets: per-trace starttime offsets in seconds (0 = aligned)."""
    combinations = [
        (i, j) for i in range(ntrc) for j in range(i + 1, ntrc)
    ]
    starttime = [UTCDateTime(2026, 1, 1) + off for off in offsets]
    corr_args = {
        "TDpreProcessing": [],
        "FDpreProcessing": [],
        "center_correlation": False,
        "normalize_correlation": True,
        "lengthToSave": length_to_save,
    }
    return {
        "corr_args": corr_args,
        "combinations": combinations,
        "starttime": starttime,
        "joint_norm": False,
    }, combinations


def run_once(nsta, fs, corr_len, length_to_save, misaligned, rng_seed=42):
    ntrc = nsta * 3
    npts = int(corr_len * fs)
    rng = np.random.default_rng(rng_seed)
    A = rng.standard_normal((ntrc, npts)).astype(np.float32)
    if misaligned:
        # every 5th trace offset by a non-integer number of samples
        offsets = [0.011 * (i // 5) for i in range(ntrc)]
    else:
        offsets = [0.0] * ntrc
    options, _ = make_options(
        ntrc, fs, corr_len, length_to_save, offsets)
    corr = build_correlator(options, fs)
    t0 = time.perf_counter()
    C, startlags = corr._pxcorr_matrix(A)
    dt = time.perf_counter() - t0
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB
    rss_mb = raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024
    return {
        "time_s": round(dt, 3),
        "peak_rss_mb": round(rss_mb, 1),
        "c_hash": hashlib.sha256(np.ascontiguousarray(C)).hexdigest(),
        "cshape": list(C.shape),
        "pairs": C.shape[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsta", type=int, default=20)
    ap.add_argument("--fs", type=float, default=50.0)
    ap.add_argument("--corr-len", type=float, default=1800.0)
    ap.add_argument("--length-to-save", type=float, default=100.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--label", default="BASELINE")
    args = ap.parse_args()

    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
        capture_output=True, text=True).stdout.strip()
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
        capture_output=True, text=True).stdout.strip()

    for scenario in ("aligned", "misaligned"):
        for rep in range(args.repeats):
            res = run_once(
                args.nsta, args.fs, args.corr_len,
                args.length_to_save, misaligned=(scenario == "misaligned"))
            rec = {
                "label": args.label,
                "branch": branch,
                "commit": sha,
                "scenario": scenario,
                "rep": rep,
                "nsta": args.nsta,
                "corr_len": args.corr_len,
                **res,
            }
            print(json.dumps(rec))
            with RESULTS.open("a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
