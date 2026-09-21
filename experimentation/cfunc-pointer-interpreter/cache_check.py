"""Item 5's decisive question, redone for `@cfunc` addresses instead of
§V's ctypes-loaded `.so` symbols: **does a cfunc address, carried as
ordinary DATA through a pointer-table argument, have the same "any kernel
referencing a ctypes symbol can never be numba-disk-cached" problem §V
found?**

Methodology matches §V exactly (EXPERIMENTS.md §V, "Confirmed against a
persistent build dir, not a tempdir artefact"): a real, non-temp directory
next to this file (`__pycache__/`, numba's own default location since
these are real importable modules -- doc 05 §4.1), one process runs first
("cold"), a SEPARATE process runs second against the SAME directory
("warm"), and `NUMBA_DEBUG_CACHE=1` is used to get numba's own cache-hit/
miss trace rather than inferring it from timing alone.

Run:
    <repo>/.venv/bin/python experimentation/cfunc-pointer-interpreter/cache_check.py cold
    <repo>/.venv/bin/python experimentation/cfunc-pointer-interpreter/cache_check.py warm
(orchestrated by `run_all.sh`, which clears __pycache__ before "cold").
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.jsonl"


def append_result(record: dict) -> None:
    record = {"experiment": "cfunc-pointer-interpreter", **record}
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "cold"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()

        from interpreter import run_program
        from nested_program import build_nested_program, param_template, make_row_regs, PTR_TABLE

        prog = build_nested_program()
        n = 1000
        rng = np.random.default_rng(1)
        high_income = rng.random(n) < 0.5
        term_cap0 = rng.uniform(6, 60, n)
        row_regs = make_row_regs(high_income, term_cap0)
        out = np.empty(n, dtype=np.float64)

        # The actual call that triggers (or reuses) compilation.
        run_program(
            prog.op, prog.step_idx, prog.arg0, prog.arg1, prog.arg2, prog.dest,
            prog.next_, prog.alt, prog.start_pc, PTR_TABLE, param_template(),
            row_regs, n, out,
        )
        elapsed = time.perf_counter() - t0

    dynamic_globals_warnings = [
        str(w.message) for w in caught
        if "cach" in str(w.message).lower() or "dynamic global" in str(w.message).lower()
    ]

    print(f"[{mode}] elapsed (import + build + first call): {elapsed*1000:.1f} ms")
    if dynamic_globals_warnings:
        print(f"[{mode}] CACHE-RELATED WARNINGS FIRED ({len(dynamic_globals_warnings)}):")
        for w in dynamic_globals_warnings:
            print(f"    {w[:200]}")
    else:
        print(f"[{mode}] no cache-related NumbaWarning fired")

    nbi_files = sorted((HERE / "__pycache__").glob("*run_program*.nbi"))
    print(f"[{mode}] run_program .nbi cache index files present: {[p.name for p in nbi_files]}")

    append_result({
        "item": "cache_check",
        "mode": mode,
        "elapsed_ms": elapsed * 1000,
        "cache_warnings": dynamic_globals_warnings,
        "nbi_files": [p.name for p in nbi_files],
    })


if __name__ == "__main__":
    main()
