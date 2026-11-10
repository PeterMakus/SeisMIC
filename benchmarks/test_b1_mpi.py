"""
MPI verification for the B1 fix (local-row allocation + Allgatherv +
per-rank CorrTrace construction without a CorrStream broadcast).

Run with:  mpirun -np <ranks> python benchmarks/test_b1_mpi.py

Checks, on every rank:
1. _pxcorr_matrix reproduces the serial baseline hash of C exactly
   (benchmarks/results.jsonl, label BASELINE-dev) under any rank count.
2. _pxcorr_inner's file partition is complete and disjoint across ranks:
   gathered trace count == number of combinations, no duplicates.
3. Each built CorrTrace's data equals the corresponding row of the
   gathered correlation matrix (the copy step preserves values).

Exits nonzero on any failure. Untracked helper for perf/b1-...;
not part of the package; do not commit without a decision.
"""
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from mpi4py import MPI
from obspy import Stream, Trace, UTCDateTime

from seismic.correlate.correlate import Correlator

NSTA, NCOMP = 20, 3
FS = 50.0
CORR_LEN = 1800.0
SAVE_LEN = 100.0
# serial baseline hashes (D1: single-precision scipy forward FFT --
# intentionally different from the pre-D baseline; the invariant is
# consistency across rank counts, not equality with pre-D output)
BASELINE = {
    "aligned": "1f8d74824ffb4effec3ec4b5ccb45d07b95b60e3aef35fdd5b925aab6a83185c",
    "misaligned": "6e584169661fde0d306282ed0bd27a1ed03429efd1d85fdeec9b31af9314db34",
}


def make_correlator(comm, misaligned):
    ntrc = NSTA * NCOMP
    combinations = [
        (i, j) for i in range(ntrc) for j in range(i + 1, ntrc)
    ]
    if misaligned:
        offsets = [0.011 * (i // 5) for i in range(ntrc)]
    else:
        offsets = [0.0] * ntrc
    corr_args = {
        "TDpreProcessing": [],
        "FDpreProcessing": [],
        "center_correlation": False,
        "normalize_correlation": True,
        "lengthToSave": SAVE_LEN,
    }
    c = Correlator.__new__(Correlator)
    c.comm = comm
    c.rank = comm.Get_rank()
    c.psize = comm.Get_size()
    c.sampling_rate = FS
    c.corr_dir = tempfile.mkdtemp()
    import logging
    c.logger = logging.getLogger("b1_mpi_test")
    logging.basicConfig(level=logging.WARNING)
    c.options = {
        "corr_args": corr_args,
        "combinations": combinations,
        "starttime": [UTCDateTime(2026, 1, 1) + off for off in offsets],
        "joint_norm": False,
        "subdivision": {"corr_len": CORR_LEN},
    }
    return c


def make_stream(misaligned):
    ntrc = NSTA * NCOMP
    st = Stream()
    for i in range(ntrc):
        tr = Trace(np.zeros(10, dtype=np.float32))
        tr.stats.network = "XX"
        tr.stats.station = "S%03d" % (i // NCOMP)
        tr.stats.channel = "BH" + "ZNE"[i % NCOMP]
        tr.stats.location = ""
        tr.stats.sampling_rate = FS
        tr.stats.starttime = UTCDateTime(2026, 1, 1) + (
            0.011 * (i // 5) if misaligned else 0.0)
        # top-level coordinates so no inventory is needed
        tr.stats.stla = 60.0 + (i // NCOMP)
        tr.stats.stlo = 10.0
        tr.stats.stel = 100.0
        st.traces.append(tr)
    return st


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    psize = comm.Get_size()
    failures = []

    for scenario in ("aligned", "misaligned"):
        misaligned = scenario == "misaligned"
        c = make_correlator(comm, misaligned)
        rng = np.random.default_rng(42)
        ntrc = NSTA * NCOMP
        npts = int(CORR_LEN * FS)
        A = rng.standard_normal((ntrc, npts)).astype(np.float32)

        # 1) kernel result identical to serial baseline on every rank
        C, startlags = c._pxcorr_matrix(A.copy())
        h = hashlib.sha256(np.ascontiguousarray(C)).hexdigest()
        if h != BASELINE[scenario]:
            failures.append(
                f"rank {rank} {scenario}: C hash {h[:12]} != baseline")
        if C.shape[0] != len(c.options["combinations"]):
            failures.append(
                f"rank {rank} {scenario}: C has {C.shape[0]} rows, "
                f"expected {len(c.options['combinations'])}")

        # 2) per-rank CorrTrace construction with a deterministic mock
        # correlation matrix (row ii holds ii*1000 + j, exactly
        # representable in float32 and asymmetric so flips are detectable)
        st = make_stream(misaligned)
        c.options["sampling_rate"] = FS
        ncomb = len(c.options["combinations"])
        C_mock = (
            np.arange(ncomb, dtype=np.float64)[:, None] * 1000.0
            + np.arange(C.shape[1], dtype=np.float64)[None, :]
        ).astype(np.float32)
        startlags_mock = np.arange(ncomb, dtype=np.float32)

        from seismic.correlate.correlate import _combined_net_stat_loc_cha
        comb = c.options["combinations"]
        row_for_id = {}
        for ii, (i0, i1) in enumerate(comb):
            h1, h2 = st[i0].stats, st[i1].stats
            flipped = (
                h1.network + h1.station + h1.channel
                > h2.network + h2.station + h2.channel)
            net, sta, loc, cha = _combined_net_stat_loc_cha(st, (i0, i1))
            row_for_id[f"{net}.{sta}.{loc}.{cha}"] = (ii, flipped)

        real_pxcorr_matrix = c._pxcorr_matrix

        def mock_pxcorr_matrix(A):
            return C_mock.copy(), startlags_mock.copy()

        c._pxcorr_matrix = mock_pxcorr_matrix
        cst = c._pxcorr_inner(st, None)
        c._pxcorr_matrix = real_pxcorr_matrix
        local = sorted(tr.id for tr in cst)
        all_ids = comm.gather(local, root=0)

        data_ok = True
        why = ""
        for tr in cst:
            ii, flipped = row_for_id[tr.id]
            expected = np.flip(C_mock[ii]) if flipped else C_mock[ii]
            if not np.array_equal(tr.data, expected):
                data_ok = False
                why = f"data mismatch on {tr.id} row {ii} flipped={flipped}"
                break
            # the row mapping via startlags must be exact as well; for
            # flipped pairs alphabetical_correlation sets
            # start_lag = -end_lag. Values pass through float32, so
            # compare with a tolerance.
            expected_lag = float(ii) if not flipped else -float(
                ii + C_mock.shape[1] / FS)
            if abs(tr.stats.start_lag - expected_lag) > 1e-3:
                data_ok = False
                why = (f"start_lag {tr.stats.start_lag} != {expected_lag} "
                       f"on {tr.id} row {ii} flipped={flipped}")
                break
        all_data_ok = comm.allreduce(data_ok, op=MPI.LAND)
        if not all_data_ok and rank == 0:
            print("debug why:", why, flush=True)

        if rank == 0:
            flat = [i for sub in all_ids for i in sub]
            if len(flat) != len(comb):
                failures.append(
                    f"{scenario}: gathered {len(flat)} traces, "
                    f"expected {len(comb)}")
            if len(set(flat)) != len(flat):
                failures.append(f"{scenario}: duplicate trace ids across ranks")
        if not all_data_ok:
            failures.append(f"rank {rank} {scenario}: trace data != C row")

    if failures:
        for f in failures:
            print(f"FAIL rank {rank}: {f}", flush=True)
        sys.exit(1)
    print(f"rank {rank}/{psize}: all checks passed", flush=True)


if __name__ == "__main__":
    main()
