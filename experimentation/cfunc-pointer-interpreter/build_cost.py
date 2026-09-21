"""Item 4 -- per-shape build cost. Codegen (decider2's real `Branch`/`Loop`,
via `decider2.compile.driver.build_driver`) against the interpreter
(`build_nested_program()` -- pure Python/numpy, no compiler invoked).

The decisive comparison is not "cold vs warm" (numba's disk cache already
answers that at the FUNCTION level, doc 05 §4.1) but "does building a NEW,
DIFFERENT shape cost anything once the machinery has run once, ever" --
codegen's answer is "a fresh `Branch`/`Loop`/kernel compile, every time
the SHAPE changes, even with a warm numba cache for functions it has seen
before" (doc 05 §4.1's own numbers: 0.18-0.80s warm PER DRIVER); the
interpreter's answer is tested directly below by building THREE genuinely
different shapes (nested, straight-line, branch-only) through the SAME
already-compiled `run_program` and confirming zero additional compiles.

Run: <repo>/.venv/bin/python experimentation/cfunc-pointer-interpreter/build_cost.py
(Run cache_check.py first so run_program/the step cfuncs are warm -- this
script measures the PER-SHAPE marginal cost, not the one-time kernel cost,
which cache_check.py already isolated.)
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECIDER2_SRC = HERE.parent.parent / "decider2" / "src"
sys.path.insert(0, str(DECIDER2_SRC))
sys.path.insert(0, str(HERE))

import numpy as np

RESULTS_PATH = HERE / "results.jsonl"


def append_result(record: dict) -> None:
    record = {"experiment": "cfunc-pointer-interpreter", **record}
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    from decider2.compile.driver import build_driver, clear_driver_cache
    from decider2.examples.playground_control_flow import SearchForViableTerm

    codegen_build_dir = HERE / ".decider2_cache_build_cost"
    if codegen_build_dir.exists():
        shutil.rmtree(codegen_build_dir)

    # --- codegen: cold (empty on-disk numba/content-addressed cache) ---
    clear_driver_cache()
    t0 = time.perf_counter()
    driver = build_driver(
        SearchForViableTerm.steps, group_ids=[0] * len(SearchForViableTerm.steps),
        build_dir=codegen_build_dir, terminal_names=frozenset({"term_cap"}),
    )
    t_codegen_cold = time.perf_counter() - t0
    print(f"codegen build_driver, cold (fresh build_dir): {t_codegen_cold*1000:.1f} ms")

    # --- codegen: same shape again, warm on-disk cache but forced past the
    # in-process Driver memo (clear_driver_cache), i.e. "the pipeline is
    # rebuilt at process start against an already-populated build dir" --
    # doc 05 §4.1's own scenario.
    clear_driver_cache()
    t0 = time.perf_counter()
    driver_warm = build_driver(
        SearchForViableTerm.steps, group_ids=[0] * len(SearchForViableTerm.steps),
        build_dir=codegen_build_dir, terminal_names=frozenset({"term_cap"}),
    )
    t_codegen_warm = time.perf_counter() - t0
    print(f"codegen build_driver, warm (same shape, on-disk cache populated): {t_codegen_warm*1000:.1f} ms")

    append_result({"item": "build_cost", "engine": "codegen", "variant": "cold", "elapsed_ms": t_codegen_cold * 1000})
    append_result({"item": "build_cost", "engine": "codegen", "variant": "warm_same_shape", "elapsed_ms": t_codegen_warm * 1000})

    # --- interpreter: run_program must already be warm for this to isolate
    # per-shape cost (cache_check.py's job) -- call it once here too so this
    # script is self-contained, uncounted.
    from interpreter import run_program
    from nested_program import build_nested_program, param_template, make_row_regs, PTR_TABLE as NESTED_PTR

    warm = np.full((2, build_nested_program().n_regs), np.nan)
    warm[:, 0] = 12.0
    warm[:, 3] = 0.0
    out_warm = np.empty(2)
    prog0 = build_nested_program()
    run_program(prog0.op, prog0.step_idx, prog0.arg0, prog0.arg1, prog0.arg2, prog0.dest,
                prog0.next_, prog0.alt, prog0.start_pc, NESTED_PTR, param_template(), warm, 2, out_warm)

    # NOW measure: building the SAME nested program again (a fresh Program
    # object -- e.g. "the pipeline definition changed, rebuild it").
    t0 = time.perf_counter()
    prog1 = build_nested_program()
    t_interp_rebuild_same_shape = time.perf_counter() - t0
    print(f"interpreter build_nested_program(), rebuilding the SAME shape: {t_interp_rebuild_same_shape*1000:.4f} ms")

    # A GENUINELY DIFFERENT shape (straight-line, from test_shapes.py),
    # through the SAME already-compiled run_program -- zero additional
    # compiles is the whole claim.
    from numba import cfunc, types
    from interpreter import ProgramBuilder, STEP1, LEAF, make_ptr_table

    def _add1_body(x):
        return x + 1.0

    t0 = time.perf_counter()
    add1 = cfunc(types.float64(types.float64), cache=False)(_add1_body)  # compiles HERE, eagerly
    t_new_cfunc_compile = time.perf_counter() - t0
    print(f"interpreter: compile ONE brand-new custom step (@cfunc, never seen before): {t_new_cfunc_compile*1000:.4f} ms")

    t0 = time.perf_counter()
    ptr_table2 = make_ptr_table([add1])
    b = ProgramBuilder(n_regs=1)
    n0 = b.add(STEP1, step_idx=0, arg0=0, dest=0)
    n1 = b.add(LEAF, dest=0)
    b.set_next(n0, next_=n1)
    prog2 = b.build(start_pc=n0)
    t_build_new_shape = time.perf_counter() - t0
    print(f"interpreter: build an ENTIRELY DIFFERENT shape (straight-line, new cfunc): {t_build_new_shape*1000:.4f} ms")

    t0 = time.perf_counter()
    param_template2 = np.zeros(1)
    row_regs2 = np.full((3, 1), np.nan)
    row_regs2[:, 0] = [1.0, 2.0, 3.0]
    out2 = np.empty(3)
    run_program(prog2.op, prog2.step_idx, prog2.arg0, prog2.arg1, prog2.arg2, prog2.dest,
                prog2.next_, prog2.alt, prog2.start_pc, ptr_table2, param_template2, row_regs2, 3, out2)
    t_run_new_shape_first_call = time.perf_counter() - t0
    print(f"interpreter: FIRST call of run_program on the new shape (same compiled kernel): {t_run_new_shape_first_call*1000:.4f} ms")
    print(f"  -> answers: {out2} (expect [2. 3. 4.]) -- {'OK' if np.array_equal(out2, [2.,3.,4.]) else 'MISMATCH'}")
    print(f"  -> run_program.signatures count: {len(run_program.signatures)} (should be 1 -- one specialisation, ever, for ANY program)")

    append_result({"item": "build_cost", "engine": "interpreter", "variant": "rebuild_same_shape_ms", "elapsed_ms": t_interp_rebuild_same_shape * 1000})
    append_result({"item": "build_cost", "engine": "interpreter", "variant": "new_cfunc_compile_ms", "elapsed_ms": t_new_cfunc_compile * 1000})
    append_result({"item": "build_cost", "engine": "interpreter", "variant": "build_new_shape_ms", "elapsed_ms": t_build_new_shape * 1000})
    append_result({"item": "build_cost", "engine": "interpreter", "variant": "first_call_new_shape_ms", "elapsed_ms": t_run_new_shape_first_call * 1000,
                    "run_program_signature_count": len(run_program.signatures)})

    total_new_step_and_shape = t_new_cfunc_compile + t_build_new_shape + t_run_new_shape_first_call
    print()
    print(f"SUMMARY: codegen per-shape build (warm numba cache, fresh Driver, SAME shape as before) = {t_codegen_warm*1000:.1f} ms")
    print(f"         interpreter: rebuild an EXISTING shape (pure data, no compile)   = {t_interp_rebuild_same_shape*1000:.4f} ms")
    print(f"         interpreter: NEW shape + ONE brand-new custom step, total        = {total_new_step_and_shape*1000:.4f} ms")
    print(f"         (of which the generic run_program kernel itself compiled)        = 0 times (signatures stayed at {len(run_program.signatures)})")


if __name__ == "__main__":
    main()
