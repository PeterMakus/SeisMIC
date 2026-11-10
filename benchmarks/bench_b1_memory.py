"""
Memory benchmark for the B1 fix: per-rank peak RSS after the correlation
kernel and after CorrTrace construction, at the current rank count.

Run with:  mpirun -np <ranks> python benchmarks/bench_b1_memory.py

Prints one JSON line per rank. Untracked helper for perf/b1-...;
not part of the package; do not commit without a decision.
"""
import json
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from mpi4py import MPI

from test_b1_mpi import make_correlator, make_stream, FS, CORR_LEN, NSTA, NCOMP


def rss_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux KiB
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    c = make_correlator(comm, False)
    rng = np.random.default_rng(42)
    A = rng.standard_normal(
        (NSTA * NCOMP, int(CORR_LEN * FS))).astype(np.float32)

    rss0 = rss_mb()
    C, startlags = c._pxcorr_matrix(A.copy())
    rss1 = rss_mb()
    del C, startlags

    st = make_stream(False)
    c.options["sampling_rate"] = FS
    cst = c._pxcorr_inner(st, None)
    rss2 = rss_mb()
    n = comm.gather(cst.count(), root=0)
    if rank == 0:
        pass
    print(json.dumps({
        "rank": rank, "psize": comm.Get_size(),
        "rss_base_mb": round(rss0, 1),
        "rss_after_kernel_mb": round(rss1, 1),
        "rss_after_corrstream_mb": round(rss2, 1),
        "local_traces": cst.count(),
        "all_traces": sum(n),
    }), flush=True)


if __name__ == "__main__":
    main()
