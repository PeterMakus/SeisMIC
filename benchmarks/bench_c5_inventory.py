"""
C5: measure the inventory-handling savings.

Builds a synthetic 100-station x 3-channel response-level inventory
(each channel carries a real 11-stage response, like the StationXML
caches Store_Client keeps) and measures:

- pickle size of the full inventory vs the response-stripped copy that
  pxcorr now bcasts when remove_response=False;
- per-trace attach_response cost (the _load_local call now skips it
  when the responses will never be used);
- the duplicate-append pathology of the old ``inv += ninv`` in
  preprocess_stream (the cached inventory grew by a duplicate Network
  copy per remote fetch).

Usage:  python benchmarks/bench_c5_inventory.py
"""
import copy
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from obspy import Stream, Trace, read_inventory
from obspy.core.inventory import (
    Channel, Inventory, Network, Station)

NSTA = 100
# a real 11-stage response (obspy bundled example, geophone EDH), cloned
# per channel so the inventory matches response-level StationXML caches
RESP_XML = ("/opt/homebrew/Caskroom/miniforge/base/envs/pyglimer-dev/lib/"
            "python3.13/site-packages/obspy/io/stationxml/data/"
            "1T_MONN_00_EDH.xml")


def make_inventory():
    """Response-level inventory: NSTA stations x 3 channels, each with a
    real 11-stage response."""
    tmpl = read_inventory(RESP_XML, level="response")
    resp = tmpl.networks[0].stations[0].channels[0].response
    nets = []
    for i in range(NSTA):
        chans = []
        for c in ("BHZ", "BHN", "BHE"):
            chans.append(Channel(
                c, "00", 0.0, 0.0, 0.0, 0.0, sample_rate=50.0,
                response=copy.deepcopy(resp)))
        sta = Station("S%03d" % i, 0.0, 0.0, 0.0, channels=chans)
        nets.append(Network("XX", stations=[sta]))
    return Inventory(nets, source="bench")


def strip(inv):
    inv = inv.copy()
    for net in inv.networks:
        for sta in net.stations:
            for cha in sta.channels:
                cha.response = None
    return inv


def main():
    inv = make_inventory()
    n_stages = len(
        inv.networks[0].stations[0].channels[0].response.response_stages)

    t0 = time.perf_counter()
    inv2 = strip(inv)
    t_strip = time.perf_counter() - t0

    n_resp = sum(
        1 for net in inv2.networks for sta in net.stations
        for cha in sta.channels if cha.response is not None)
    n_resp0 = sum(
        1 for net in inv.networks for sta in net.stations
        for cha in sta.channels if cha.response is not None)

    p_full = len(pickle.dumps(inv))
    p_strip = len(pickle.dumps(inv2))

    t0 = time.perf_counter()
    copy.deepcopy(inv)
    t_deep = time.perf_counter() - t0

    # attach_response cost per trace batch (what _load_local skips now)
    st = Stream()
    for i in range(60):
        tr = Trace(np.zeros(90001, dtype=np.float32))
        tr.stats.network = "XX"
        tr.stats.station = "S%03d" % (i // 3)
        tr.stats.channel = "BH" + "ZNE"[i % 3]
        st.append(tr)
    t0 = time.perf_counter()
    st.attach_response(inv)
    t_attach = time.perf_counter() - t0

    # old duplicate-append: each remote fetch appended the network again
    inv_cached = make_inventory()
    ninv = inv_cached.select(network="XX", station="S000")
    before = len(inv_cached.networks)
    inv_cached += ninv   # what _write_inventory does
    inv_cached += ninv   # what the old preprocess_stream line repeated
    after = len(inv_cached.networks)

    print("inventory: %d stations x 3 channels, %d-stage responses"
          % (NSTA, n_stages))
    print("responses: full %d / stripped %d (strip+copy %.3f s)"
          % (n_resp0, n_resp, t_strip))
    print("pickle size: full %.2f MB -> stripped %.2f MB (bcast payload "
          "per rank)" % (p_full / 1e6, p_strip / 1e6))
    print("attach_response on 60 traces: %.1f ms (skipped when "
          "remove_response=False)" % (t_attach * 1e3))
    print("old duplicate append: networks %d -> %d after 2 remote "
          "fetches of 1 network" % (before, after))


if __name__ == "__main__":
    main()
