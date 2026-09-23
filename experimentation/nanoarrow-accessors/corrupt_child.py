"""One (case, approach) run. Prints RESULT <list> on success; raises on
refusal; may crash. Exit code and output are classified by corrupt.py."""
from __future__ import annotations

import ctypes
import sys

import numpy as np
import polars as pl

import nashim  # noqa: F401
import arrowc
from kernel import CONTAINS, ERR_LEAF, LEAF, STR, build_tree, run_tree
from kernel_b import run_tree_b
from kernel_c import run_tree_c
from nashim import FULL, NanoView, lib

STRINGS = ["dog", "a much longer merchant descriptor dog", "cat", None, "x" * 40 + "dog", "hotdog"]
ROW = 1  # a long (non-inline) string: its view is (len, prefix, buffer_index, offset)
TREE = build_tree([(STR, 0, CONTAINS, 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])


def poke_u32(addr, value):
    ctypes.c_uint32.from_address(addr).value = value


def poke_i64(addr, value):
    ctypes.c_int64.from_address(addr).value = value


def corrupt(case, arr: arrowc.ArrowArray):
    """Mutate the exported Arrow data in place. `arr` is the ctypes ArrowArray."""
    views = arr.buffers[1]
    at = views + 16 * (ROW + arr.offset)
    if case == "clean":
        return
    if case == "bad_buffer_index":
        poke_u32(at + 8, 1000)                    # variadic buffer 1000 of 1
    elif case == "bad_offset":
        poke_u32(at + 12, 2**31 - 1)              # far past the data buffer
    elif case in ("bad_length", "bad_length_scan"):
        poke_u32(at + 0, 2**30)                   # overruns the buffer by 1 GB
    elif case == "null_validity_nonzero_null_count":
        # build from a no-null series (validity buffer is NULL), then claim nulls
        assert arr.buffers[0] is None or arr.buffers[0] == 0
        arr.null_count = 3
    elif case == "n_buffers_truncated":
        arr.n_buffers = 2
    elif case == "lying_sizes_buffer":
        # the producer's variadic sizes buffer claims 1 TB, and the offset points
        # past the real buffer but inside the claimed size
        sizes = arr.buffers[arr.n_buffers - 1]
        poke_i64(sizes, 2**40)
        poke_u32(at + 12, 2**26)                  # 64 MB past the real ~90-byte buffer
    else:
        raise ValueError(case)


def main():
    case, approach = sys.argv[1], sys.argv[2]
    # bad_length_scan: a pattern that is NOT in the string, so CONTAINS must scan
    # the whole (claimed) length instead of stopping at the first match
    pattern = "zzz" if case == "bad_length_scan" else "dog"
    if case == "sliced_offset":
        base = pl.Series("s", ["pad0", "pad1", "pad2"] + STRINGS + ["pad3"])
        s = base.slice(3, len(STRINGS))           # zero-copy: Arrow offset 3
    elif case == "null_validity_nonzero_null_count":
        s = pl.Series("s", [x or "none" for x in STRINGS])
    else:
        s = pl.Series("s", STRINGS)
    n = len(s)
    feats = np.empty((n, 0))
    if approach == "A":
        v = arrowc.export(s)
        c = v.chunks[0]
        if case not in ("sliced_offset",):
            corrupt(case, c._arr)
            # arrowc snapshotted its fields at export; re-read them from the (now
            # corrupted) ArrowArray exactly as a fresh export would, so A trusts
            # the same producer metadata B and C re-import
            c.null_count = c._arr.null_count
            nb = c._arr.n_buffers
            n_data = nb - 3
            if n_data <= 0:
                c.data, c.data_sizes = [], []
            else:
                c.data = [c._arr.buffers[2 + k] for k in range(n_data)]
                c.data_sizes = list((ctypes.c_int64 * n_data).from_address(c._arr.buffers[nb - 1]))
        out = run_tree(feats, [v], TREE, [pattern])
    else:
        nv = NanoView(s)
        if case not in ("sliced_offset",):
            corrupt(case, nv.chunks[0].array)
            # re-import from the (now corrupted) ArrowArray, as if it arrived so
            lib.sm_view_reset(nv.chunks[0].view)
            rc = lib.sm_view_set(nv.chunks[0].view, ctypes.byref(nv.schema), ctypes.byref(nv.chunks[0].array), nv.err)
            if rc != 0:
                raise nashim.NanoError(f"import refused rc={rc}: {lib.sm_error_message(nv.err).decode()}")
            nv.chunks[0].arrowc_chunk = None
        if approach == "B":
            out = run_tree_b(feats, [nv], TREE, [pattern])
        elif approach == "B-checked":
            out = run_tree_b(feats, [nv], TREE, [pattern], checked=True)
        elif approach == "C-full":
            out = run_tree_c(feats, [nv], TREE, [pattern], level=FULL)
        elif approach == "C-trusting":
            out = run_tree_c(feats, [nv], TREE, [pattern], level=FULL, trusting=True)
        else:
            raise ValueError(approach)
    print("RESULT", out.tolist(), flush=True)


if __name__ == "__main__":
    main()
