"""Q1: is polars -> Arrow zero-copy for a String column on polars 1.41?

Proof standard: (a) identical buffer addresses across two calls, (b) O(1)
cost as rows grow, against `_get_buffers()` which is known to be O(rows).
Writes one JSON line per measurement to results.jsonl.
"""
from __future__ import annotations

import ctypes
import json
import sys
import time

import numpy as np
import polars as pl

import arrowc

OUT = open("results.jsonl", "a")


def log(**rec):
    rec["probe"] = "zero_copy"
    OUT.write(json.dumps(rec) + "\n")
    OUT.flush()
    print(json.dumps(rec))


def make(n: int, kind: str) -> pl.Series:
    rng = np.random.default_rng(0)
    if kind == "short":  # all <= 12 bytes: inline in the view
        pool = ["dog", "cat", "fish", "employer-x", "a", "twelve-chars"]
        return pl.Series("s", [pool[i] for i in rng.integers(0, len(pool), n)])
    # mixed: half long (>12) so the variadic data buffers are exercised
    pool = ["dog", "cat", "a much longer merchant descriptor", "TRANSFER FROM 1234567890", "fish"]
    return pl.Series("s", [pool[i] for i in rng.integers(0, len(pool), n)])


def best_of(fn, reps=7):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t0)
        del r
    return min(ts)


def main():
    for n in (100_000, 1_000_000, 4_000_000):
        for kind in ("short", "mixed"):
            s = make(n, kind)
            assert s.n_chunks() == 1

            # --- Arrow C stream (PyCapsule interface) -------------------------
            v1 = arrowc.export(s)
            v2 = arrowc.export(s)
            same = (v1.chunks[0].views == v2.chunks[0].views and v1.chunks[0].data == v2.chunks[0].data
                    and v1.chunks[0].validity == v2.chunks[0].validity)
            v1.release(); v2.release()
            t_stream = best_of(lambda: arrowc.export(s))
            log(route="__arrow_c_stream__", n=n, kind=kind, same_address_twice=same,
                seconds=t_stream, ns_per_row=t_stream / n * 1e9, format=v1.format,
                n_data_buffers=len(v1.chunks[0].data))

            # --- _export_arrow_to_c (single chunk) ------------------------------
            def via_export():
                arr = arrowc.ArrowArray(); sch = arrowc.ArrowSchema()
                s._export_arrow_to_c(ctypes.addressof(arr), ctypes.addressof(sch))
                return arr, sch
            a1, sc1 = via_export(); a2, sc2 = via_export()
            same2 = a1.buffers[1] == a2.buffers[1] and a1.buffers[1] == v1.chunks[0].views
            for a in (a1, a2):
                a.release(ctypes.byref(a))
            for sc in (sc1, sc2):
                sc.release(ctypes.byref(sc))
            def timed_export():
                a, sc = via_export()
                a.release(ctypes.byref(a)); sc.release(ctypes.byref(sc))
            t_exp = best_of(timed_export)
            log(route="_export_arrow_to_c", n=n, kind=kind, same_address_twice=same2,
                same_as_stream_route=bool(same2), seconds=t_exp, ns_per_row=t_exp / n * 1e9)

            # --- _get_buffers (the known-copying route) -------------------------
            def gb():
                b = s._get_buffers()
                return b["offsets"]._get_buffer_info()[0], b["values"]._get_buffer_info()[0]
            p1 = gb(); p2 = gb()
            t_gb = best_of(lambda: s._get_buffers())
            log(route="_get_buffers", n=n, kind=kind, same_address_twice=(p1 == p2),
                seconds=t_gb, ns_per_row=t_gb / n * 1e9)

            # --- _get_buffer_info directly on the String series ------------------
            try:
                s._get_buffer_info()
                log(route="_get_buffer_info", n=n, kind=kind, works=True)
            except Exception as exc:
                if n == 100_000 and kind == "short":
                    log(route="_get_buffer_info", n=n, kind=kind, works=False, error=repr(exc))

            # --- requesting large_utf8 through the stream: does polars convert? --
            # (requested_schema must be an ArrowSchema PyCapsule; we build one by
            #  hand with format "U" to ask for large_utf8.)
            if n == 1_000_000 and kind == "mixed":
                try:
                    sch = arrowc.ArrowSchema()
                    sch.format = b"U"; sch.name = b"s"; sch.metadata = None; sch.flags = 2
                    sch.n_children = 0
                    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
                    PyCapsule_New.restype = ctypes.py_object
                    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
                    cap = PyCapsule_New(ctypes.addressof(sch), b"arrow_schema", None)
                    vU = arrowc.export(s, cap)
                    t_U = best_of(lambda: arrowc.export(s, cap), reps=3)
                    log(route="__arrow_c_stream__ requested large_utf8", n=n, kind=kind,
                        format_returned=vU.format, seconds=t_U, ns_per_row=t_U / n * 1e9)
                except Exception as exc:
                    log(route="__arrow_c_stream__ requested large_utf8", n=n, kind=kind, error=repr(exc))

            # --- multi-chunk series -------------------------------------------
            if n == 1_000_000 and kind == "mixed":
                s2 = pl.concat([s[: n // 2], s[n // 2:]])
                vv = arrowc.export(s2)
                log(route="__arrow_c_stream__ two chunks", n=n, kind=kind, polars_n_chunks=s2.n_chunks(),
                    exported_chunks=[c.length for c in vv.chunks],
                    get_buffers_on_chunked=_get_buffers_chunked(s2))
            del s

    OUT.close()


def _get_buffers_chunked(s2):
    try:
        b = s2._get_buffers()
        return {"ok": True, "offsets_len": len(b["offsets"]), "values_len": len(b["values"]),
                "n_chunks_of_offsets": b["offsets"].n_chunks()}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


if __name__ == "__main__":
    main()
