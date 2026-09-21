"""Subprocess worker for the panic-safety demonstration (item 5). Run in
its OWN process (see `panic_demo.py`) because the unprotected modes are
expected to kill the process — that is the entire point of the demo, and
must not take the measurement harness down with it.

Modes:
  trivial_unprotected  -- `deliberately_panic_unprotected(1)`, called from
                          inside njit. No catch_unwind on the Rust side.
  trivial_protected    -- same deliberate panic, through
                          `deliberately_panic_protected`, which
                          catch_unwind-wraps the panic and returns NaN.
  tree_unprotected     -- a REALISTIC trigger: the credit-kernel shape,
                          with `left[0]` corrupted to an out-of-range node
                          index, walked via `walk_row_unprotected`. The
                          resulting out-of-bounds SLICE access panics
                          inside Rust (bounds-checked indexing, not raw
                          pointer arithmetic — see rust_cabi_tree/src/
                          lib.rs), same failure mode as the trivial case
                          but reached through the actual tree walker
                          `bench_perf.py` measures, not a synthetic stub.
  tree_protected       -- same corrupted tree, via `walk_row_cabi`
                          (catch_unwind-protected): returns NaN, and this
                          script THEN makes a second, uncorrupted call to
                          prove the process is still alive and correct,
                          not just "didn't crash on this one call".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tree-codegen-vs-interpreted"))

import tree_shapes as ts  # noqa: E402
from tree_walk_cabi import (  # noqa: E402
    deliberately_panic_protected,
    deliberately_panic_unprotected,
    walk_row_cabi,
    walk_row_cabi_unprotected,
)


def corrupted_tree():
    shape = ts.build_full_binary(depth=2, n_features=2, seed=1)
    flat = ts.to_flat_tree(shape)
    # deliberate corruption: node 0's "then" child is set to an index far
    # outside every tree array's real length (7 nodes for depth 2) — the
    # kind a bit-flip, a bad deserialisation, or a hostile input crafting
    # its way past validation could produce, and exactly the shape the
    # brief calls out ("a segfault inside a bank's credit kernel").
    flat.left[0] = 999_999
    row = np.array([50.0, 50.0])
    str_row = np.zeros(0, dtype=np.int32)
    return flat, row, str_row


def main() -> None:
    mode = sys.argv[1]
    print(f"[worker pid={__import__('os').getpid()}] mode={mode}", flush=True)

    if mode == "trivial_unprotected":
        print("calling deliberately_panic_unprotected(1)...", flush=True)
        result = deliberately_panic_unprotected(1)
        print(f"UNEXPECTED: returned normally with {result}", flush=True)

    elif mode == "trivial_protected":
        print("calling deliberately_panic_protected(1)...", flush=True)
        result = deliberately_panic_protected(1)
        print(f"returned cleanly: {result} (nan={result != result})", flush=True)
        result_ok = deliberately_panic_protected(0)
        print(f"subsequent good call still works: {result_ok}", flush=True)

    elif mode == "tree_unprotected":
        flat, row, str_row = corrupted_tree()
        print("calling walk_row_cabi_unprotected on a corrupted tree (left[0]=999999)...", flush=True)
        result = walk_row_cabi_unprotected(
            flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
            flat.pat_start, flat.pat_count, flat.patterns,
            flat.left, flat.right, flat.leaf_value, row, str_row,
        )
        print(f"UNEXPECTED: returned normally with {result}", flush=True)

    elif mode == "tree_protected":
        flat, row, str_row = corrupted_tree()
        print("calling walk_row_cabi (catch_unwind-protected) on the SAME corrupted tree...", flush=True)
        result = walk_row_cabi(
            flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
            flat.pat_start, flat.pat_count, flat.patterns,
            flat.left, flat.right, flat.leaf_value, row, str_row,
        )
        print(f"returned cleanly: {result} (nan={result != result})", flush=True)

        # prove the process survived and is still correct: an uncorrupted
        # tree, same process, same loaded library, right after the panic.
        good_shape = ts.build_full_binary(depth=2, n_features=2, seed=1)
        good_flat = ts.to_flat_tree(good_shape)
        good_result = walk_row_cabi(
            good_flat.kind, good_flat.feat_idx, good_flat.op_code, good_flat.thresh,
            good_flat.pat_start, good_flat.pat_count, good_flat.patterns,
            good_flat.left, good_flat.right, good_flat.leaf_value, row, str_row,
        )
        print(f"subsequent call on an UNCORRUPTED tree, same process: {good_result}", flush=True)

    else:
        raise SystemExit(f"unknown mode {mode!r}")

    print("worker exiting normally (returncode 0)", flush=True)


if __name__ == "__main__":
    main()
