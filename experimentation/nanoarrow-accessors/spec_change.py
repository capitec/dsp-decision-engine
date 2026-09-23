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


def try_(label, fn):
    try:
        out = fn()
        ok = out == EXPECT
        outcome = "correct" if ok else "wrong_answer"
        detail = str(out)
    except Exception as e:  # noqa: BLE001
        outcome = "refused"
        detail = f"{type(e).__name__}: {str(e)[:140]}"
    return outcome, detail


def run_case(fmt, sv_a, nv_factory):
    feats = np.empty((len(STRINGS), 0))
    cases = {
        "A": lambda: run_tree(feats, [sv_a], TREE, ["dog"]).tolist(),
        "B": lambda: run_tree_b(feats, [nv_factory()], TREE, ["dog"]).tolist(),
        "B-checked": lambda: run_tree_b(feats, [nv_factory()], TREE, ["dog"], checked=True).tolist(),
        "C-full": lambda: run_tree_c(feats, [nv_factory()], TREE, ["dog"], level=FULL).tolist(),
    }
    for ap, fn in cases.items():
        outcome, detail = try_(ap, fn)
        record(probe="spec_change", format=fmt, approach=ap, outcome=outcome, detail=detail)
        print(f"format {fmt!r:6} {ap:<10} {outcome:<13} {detail}", flush=True)


def main():
    for fmt in ("u", "U"):
        schema, arr, sv_a = build_offsets_array(STRINGS, fmt)
        run_case(fmt, sv_a, lambda: NanoView(structs=(schema, arr)))
    # a made-up future format: what does each side say?
    schema, arr, sv_a = build_offsets_array(STRINGS, "u")
    schema.format = b"vz"; sv_a.format = "vz"
    run_case("vz", sv_a, lambda: NanoView(structs=(schema, arr)))
    # a real polars column whose dtype is Null (pl.Series([None]) infers Null, not String)
    s = pl.Series("s", [None] * len(STRINGS))
    global EXPECT
    EXPECT = [0] * len(STRINGS)
    def a_null():
        v = arrowc.export(s)
        return run_tree(np.empty((len(STRINGS), 0)), [v], TREE, ["dog"]).tolist()
    outcome, detail = try_("A", a_null)
    record(probe="spec_change", format="n (polars Null dtype)", approach="A", outcome=outcome, detail=detail)
    print(f"format 'n'    A          {outcome:<13} {detail}")
    feats = np.empty((len(STRINGS), 0))
    for ap, fn in {"B": lambda: run_tree_b(feats, [NanoView(s)], TREE, ["dog"]).tolist(),
                   "B-checked": lambda: run_tree_b(feats, [NanoView(s)], TREE, ["dog"], checked=True).tolist(),
                   "C-full": lambda: run_tree_c(feats, [NanoView(s)], TREE, ["dog"], level=FULL).tolist()}.items():
        outcome, detail = try_(ap, fn)
        record(probe="spec_change", format="n (polars Null dtype)", approach=ap, outcome=outcome, detail=detail)
        print(f"format 'n'    {ap:<10} {outcome:<13} {detail}")


if __name__ == "__main__":
    main()
