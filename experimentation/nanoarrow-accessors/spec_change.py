"""What each approach does when the producer's format is not polars' 'vu':
hand-built Arrow arrays in 'u' (utf8, int32 offsets) and 'U' (large_utf8,
int64 offsets), a real polars Null column ('n'), and a made-up future
format ('vz'). No pyarrow is installed; the arrays are built with ctypes
exactly as another Arrow producer would hand them over."""
from __future__ import annotations

import ctypes
import traceback

import numpy as np
import polars as pl

import nashim
import arrowc
from kernel import CONTAINS, LEAF, STR, build_tree, run_tree
from kernel_b import run_tree_b
from kernel_c import run_tree_c
from nashim import FULL, NanoView
from results_io import record

TREE = build_tree([(STR, 0, CONTAINS, 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])
STRINGS = ["dog", "a much longer merchant descriptor dog", "cat", None, "x" * 40 + "dog", "hotdog"]
EXPECT = [1, 1, 0, 0, 1, 1]
_keep = []  # keep ctypes callbacks and numpy buffers alive

_SchemaRelease = arrowc.ArrowSchema._fields_[7][1]
_ArrayRelease = arrowc.ArrowArray._fields_[8][1]


def _schema_release(p):
    p.contents.release = ctypes.cast(None, _SchemaRelease)


def _array_release(p):
    p.contents.release = ctypes.cast(None, _ArrayRelease)


def build_offsets_array(strings, fmt):
    """An ArrowSchema+ArrowArray pair for 'u' (int32 offsets) or 'U' (int64)."""
    enc = [s.encode() if s is not None else b"" for s in strings]
    off_dtype = np.int32 if fmt == "u" else np.int64
    offsets = np.zeros(len(enc) + 1, dtype=off_dtype)
    offsets[1:] = np.cumsum([len(b) for b in enc])
    data = np.frombuffer(b"".join(enc) or b"\0", dtype=np.uint8).copy()
    n_null = sum(s is None for s in strings)
    validity = np.zeros((len(enc) + 7) // 8, dtype=np.uint8)
    for i, s in enumerate(strings):
        if s is not None:
            validity[i >> 3] |= 1 << (i & 7)
    schema = arrowc.ArrowSchema()
    schema.format = fmt.encode(); schema.name = b"s"; schema.metadata = None; schema.flags = 2
    schema.n_children = 0; schema.children = None; schema.dictionary = None
    sr = _SchemaRelease(_schema_release); schema.release = sr; schema.private_data = None
    arr = arrowc.ArrowArray()
    arr.length = len(enc); arr.null_count = n_null; arr.offset = 0; arr.n_buffers = 3; arr.n_children = 0
    bufs = (ctypes.c_void_p * 3)(validity.ctypes.data if n_null else None, offsets.ctypes.data, data.ctypes.data)
    arr.buffers = ctypes.cast(bufs, ctypes.POINTER(ctypes.c_void_p)); arr.children = None; arr.dictionary = None
    ar = _ArrayRelease(_array_release); arr.release = ar; arr.private_data = None
    _keep.extend([offsets, data, validity, bufs, sr, ar])
    chunk = arrowc.Chunk(length=arr.length, offset=0, null_count=n_null, validity=bufs[0] or 0,
                         views=bufs[1], data=[bufs[2]], data_sizes=[len(data)], _arr=None)
    return schema, arr, arrowc.StringView(format=fmt, chunks=[chunk])



def make(fmt):
    """(sv_for_A, nanoview_factory_for_B/C, expect) for a format label."""
    global EXPECT
    if fmt in ("u", "U"):
        schema, arr, sv = build_offsets_array(STRINGS, fmt)
        return sv, (lambda: NanoView(structs=(schema, arr))), EXPECT
    if fmt == "vz":   # a REAL format (binary_view): the schema lies about utf8 buffers
        schema, arr, sv = build_offsets_array(STRINGS, "u"); schema.format = b"vz"; sv.format = "vz"
        return sv, (lambda: NanoView(structs=(schema, arr))), EXPECT
    if fmt == "q":    # a format nanoarrow 0.9.0 does not know
        schema, arr, sv = build_offsets_array(STRINGS, "u"); schema.format = b"q"; sv.format = "q"
        return sv, (lambda: NanoView(structs=(schema, arr))), EXPECT
    if fmt == "n":    # a real polars column whose dtype is Null (pl.Series([None]) infers Null)
        s = pl.Series("s", [None] * len(STRINGS))
        return (lambda: arrowc.export(s)), (lambda: NanoView(s)), [0] * len(STRINGS)
    raise ValueError(fmt)


LABELS = {"u": "u (utf8, int32 offsets)", "U": "U (large_utf8, int64 offsets)",
          "vz": "vz (schema says binary_view, buffers are utf8)", "q": "q (unknown format)",
          "n": "n (polars Null dtype column)"}


def child(fmt, ap):
    sv_a, nv, expect = make(fmt)
    feats = np.empty((len(STRINGS), 0))
    if ap == "A":
        v = sv_a() if callable(sv_a) else sv_a
        out = run_tree(feats, [v], TREE, ["dog"])
    elif ap == "B":
        out = run_tree_b(feats, [nv()], TREE, ["dog"])
    elif ap == "B-checked":
        out = run_tree_b(feats, [nv()], TREE, ["dog"], checked=True)
    else:
        out = run_tree_c(feats, [nv()], TREE, ["dog"], level=FULL)
    print("RESULT", out.tolist(), "EXPECT", expect, flush=True)


def main():
    import os, subprocess, sys
    if len(sys.argv) == 3:
        child(sys.argv[1], sys.argv[2]); return
    here = os.path.dirname(os.path.abspath(__file__))
    for fmt in ("u", "U", "n", "q", "vz"):
        for ap in ("A", "B", "B-checked", "C-full"):
            p = subprocess.run([sys.executable, os.path.join(here, "spec_change.py"), fmt, ap],
                               capture_output=True, text=True, cwd=here, timeout=120)
            res = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
            if p.returncode < 0 or p.returncode >= 128:
                outcome, detail = "crash", f"signal {-p.returncode if p.returncode < 0 else p.returncode - 128}"
            elif p.returncode != 0:
                last = [ln for ln in p.stderr.strip().splitlines() if ln.strip()][-1:] or ["?"]
                outcome, detail = "refused", last[0][:150]
            else:
                got, expect = res[0].split(" EXPECT ")
                got = got[len("RESULT "):]
                if got == expect:
                    outcome, detail = "correct", got
                elif "-1" in got:
                    outcome, detail = "error_leaf", got
                else:
                    outcome, detail = "wrong_answer", got
            record(probe="spec_change", format=LABELS[fmt], approach=ap, returncode=p.returncode, outcome=outcome, detail=detail)
            print(f"{LABELS[fmt]:<48} {ap:<10} rc={p.returncode:>4}  {outcome:<12} {detail}", flush=True)


if __name__ == "__main__":
    main()
