"""Supplementary check: does the ns/row crossover seen approaching depth 9
(see RESULTS.md §4) continue at depth 10? Not one of the four required
shapes — run separately, on request, because a depth-10 full-binary tree's
cold compile is ~20s+ and there's no reason to pay that on every run of
`run_experiment.py`.

Same methodology as the main run: real `decider2.trees.codegen` +
`compile.cache` path, explicit-signature njit compile, true cold cache
(wipes `codegen_bench.BUILD_DIR` first), min-of-7 ns/row, and an identical-
answers assertion against the interpreted kernel before any timing counts.

Run: <repo>/.venv/bin/python supplementary_depth10.py
"""
from __future__ import annotations

import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import tree_shapes as ts  # noqa: E402
import interpreted_kernel as ik  # noqa: E402
import codegen_bench as cb  # noqa: E402
from run_experiment import (  # noqa: E402
    N_ROWS, append_result, build_codegen_matrices, build_interpreted_matrices,
    check_memory, cold_start_total, min_ns_per_row, warm_up_interpreted_kernel_once,
    warm_up_numba_process,
)


def main() -> None:
    check_memory()
    shutil.rmtree(cb.BUILD_DIR, ignore_errors=True)
    warm_up_numba_process()
    warm_up_interpreted_kernel_once()

    shape = ts.build_full_binary(10, seed=1)
    print(f"full_binary_d10: leaves={shape.leaf_count} nodes={shape.node_count}")

    # process-stable seed — see run_experiment.py's comment on this same fix
    numeric, string_codes = ts.make_rows(shape, N_ROWS, seed=zlib.crc32(shape.name.encode()) & 0xFFFF)

    t0 = time.perf_counter()
    codegen = cb.compile_codegen_tree(shape)
    print(f"cold compile wall: {time.perf_counter()-t0:.2f} s "
          f"(emitted {codegen.emitted_lines} lines)")

    feat_matrix, str_matrix = build_codegen_matrices(shape, codegen.emitted, numeric, string_codes)
    codegen_out = np.zeros(N_ROWS)
    codegen.row_loop(feat_matrix, str_matrix, codegen_out)
    codegen_ns_per_row = min_ns_per_row(codegen.row_loop, (feat_matrix, str_matrix, codegen_out), N_ROWS)

    t0 = time.perf_counter()
    flat = ts.to_flat_tree(shape)
    flatten_s = time.perf_counter() - t0
    num_matrix, str_matrix2 = build_interpreted_matrices(flat, numeric, string_codes)
    interp_out = np.zeros(N_ROWS)
    interp_args = (num_matrix, str_matrix2, flat.kind, flat.feat_idx, flat.op_code, flat.thresh,
                    flat.pat_start, flat.pat_count, flat.patterns, flat.left, flat.right,
                    flat.leaf_value, interp_out)
    ik.walk_batch(*interp_args)
    interp_ns_per_row = min_ns_per_row(ik.walk_batch, interp_args, N_ROWS)

    assert np.array_equal(codegen_out, interp_out), "codegen and interpreted disagree"
    print("answers identical: OK")
    print(f"codegen ns/row: {codegen_ns_per_row:.1f}")
    print(f"interpreted ns/row: {interp_ns_per_row:.1f}")
    print(f"interpreted flatten: {flatten_s*1e6:.1f} us")

    append_result({
        "shape": shape.name,
        "supplementary": True,
        "leaf_count": shape.leaf_count,
        "node_count": shape.node_count,
        "n_rows": N_ROWS,
        "emitted_lines": codegen.emitted_lines,
        "within_production_line_cap_500": codegen.emitted_lines <= 500,
        "codegen": {
            "timings_s": codegen.timings,
            "cold_start_total_s": cold_start_total(codegen.timings),
            "ns_per_row": codegen_ns_per_row,
        },
        "interpreted": {"flatten_s": flatten_s, "ns_per_row": interp_ns_per_row},
        "answers_identical": True,
    })


if __name__ == "__main__":
    main()
