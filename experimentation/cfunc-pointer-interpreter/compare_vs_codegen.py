"""Items 3, 5 and 6 -- identical work, decider2's real `Branch`/`Loop`
codegen against the cfunc-pointer interpreter, on
`decider2.examples.playground_control_flow.SearchForViableTerm`
(`Loop(Branch(steps1, Loop(steps2)))`, real, unmodified, imported from
decider2/src -- not reimplemented).

Item 3: 100k rows, answers asserted `np.array_equal` (not `isclose`),
ns/row for both engines, one measurement per process invocation -- run
this script several times (`run_all.sh` does 5) and read the RANGE across
runs from results.jsonl, not just one run's mean (§Q: "the finding that
matters is variance, not the mean").

Item 5: `build_driver`'s own segment count and `type(kernel_fn)` -- the
same structural check §V ran, against decider2's REAL compiler.

Item 6: retune `term_still_short`'s `floor` param (40.0 -> 25.0) for BOTH
engines. Codegen: change the `ResolvedParams` value passed to the SAME
`Driver`/`CompiledSegment` -- `driver.signatures` length must not grow.
Interpreter: change ONE float in `param_template` -- `run_program.
signatures` length must not grow either (it is one generic kernel; a value
change cannot possibly retrigger numba's type-based specialisation, but
this asserts it rather than assuming it).

Run: <repo>/.venv/bin/python experimentation/cfunc-pointer-interpreter/compare_vs_codegen.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECIDER2_SRC = HERE.parent.parent / "decider2" / "src"
sys.path.insert(0, str(DECIDER2_SRC))
sys.path.insert(0, str(HERE))

import numpy as np

RESULTS_PATH = HERE / "results.jsonl"
N_ROWS = 100_000
REPS = 5


def append_result(record: dict) -> None:
    record = {"experiment": "cfunc-pointer-interpreter", **record}
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def make_rows(n: int, seed: int):
    # §Q's lesson: never seed from hash() of a string (CPython randomises
    # per process). A fixed integer seed via np.random.default_rng is
    # reproducible across processes and runs -- exactly what "range across
    # SEVERAL runs" needs to mean something.
    rng = np.random.default_rng(seed)
    high_income = rng.random(n) < 0.5
    term_cap0 = rng.uniform(6.0, 60.0, n)
    return high_income, term_cap0


def main() -> None:
    from decider2.compile.driver import build_driver, ResolvedParams, CompiledSegment
    from decider2.examples.playground_control_flow import SearchForViableTerm

    from interpreter import run_program
    from nested_program import build_nested_program, param_template, make_row_regs, PTR_TABLE

    build_dir = HERE / ".decider2_cache_compare"
    driver = build_driver(
        SearchForViableTerm.steps, group_ids=[0] * len(SearchForViableTerm.steps),
        build_dir=build_dir, terminal_names=frozenset({"term_cap"}),
    )

    # --- item 5: structural check, decider2's own build_driver ---
    seg = driver.segments[0]
    print(f"[item5] segment count: {len(driver.segments)}")
    print(f"[item5] segment kind: {seg.kind!r} ({type(seg).__name__})")
    print(f"[item5] kernel_fn type: {type(seg.kernel_fn)}")
    print(f"[item5] is CompiledSegment (not FallbackSegment): {isinstance(seg, CompiledSegment)}")
    assert len(driver.segments) == 1, "expected the whole nested Loop(Branch(...Loop...)) to fuse into ONE segment"
    assert isinstance(seg, CompiledSegment)

    step_name = SearchForViableTerm.steps[0].name  # "term_cap"
    default_scalar = {
        (step_name, "term_still_short__floor"): 40.0,
        (step_name, "adjustment_strategy__fast_track_bump__fast_track_should_continue__micro_steps"): 3.0,
        (step_name, "adjustment_strategy__fast_track_bump__fast_bump__jump"): 4.0,
        (step_name, "adjustment_strategy__slow_bump__step_size"): 3.0,
    }
    resolved = ResolvedParams(per_step_scalar=default_scalar, per_step_bundle={}, shared=None)

    # --- correctness + timing, item 3 ---
    high_income, term_cap0 = make_rows(N_ROWS, seed=20260921)

    prog = build_nested_program()
    row_regs = make_row_regs(high_income, term_cap0)
    tmpl = param_template()

    # warm-up (compiles on first call if not already warm; not timed)
    registry = {"high_income": high_income.copy(), "term_cap": term_cap0.copy()}
    seg.run(registry, resolved, N_ROWS)
    out_codegen = registry["term_cap"].copy()

    out_interp = np.empty(N_ROWS, dtype=np.float64)
    run_program(prog.op, prog.step_idx, prog.arg0, prog.arg1, prog.arg2, prog.dest,
                prog.next_, prog.alt, prog.start_pc, PTR_TABLE, tmpl, row_regs, N_ROWS, out_interp)

    n_mismatch = np.sum(~np.isclose(out_codegen, out_interp))
    max_abs_diff = np.max(np.abs(out_codegen - out_interp))
    exact_equal = np.array_equal(out_codegen, out_interp)
    print(f"[item3] exact equal (np.array_equal): {exact_equal}")
    print(f"[item3] mismatched rows (tolerance): {n_mismatch}/{N_ROWS}, max abs diff: {max_abs_diff}")
    assert exact_equal, f"codegen and interpreter disagree on {n_mismatch} rows, max diff {max_abs_diff}"

    # timing -- codegen
    codegen_times = []
    for _ in range(REPS):
        registry = {"high_income": high_income.copy(), "term_cap": term_cap0.copy()}
        t0 = time.perf_counter()
        seg.run(registry, resolved, N_ROWS)
        codegen_times.append(time.perf_counter() - t0)
    codegen_ns_row = [t / N_ROWS * 1e9 for t in codegen_times]

    # timing -- interpreter
    interp_times = []
    for _ in range(REPS):
        out = np.empty(N_ROWS, dtype=np.float64)
        t0 = time.perf_counter()
        run_program(prog.op, prog.step_idx, prog.arg0, prog.arg1, prog.arg2, prog.dest,
                    prog.next_, prog.alt, prog.start_pc, PTR_TABLE, tmpl, row_regs, N_ROWS, out)
        interp_times.append(time.perf_counter() - t0)
    interp_ns_row = [t / N_ROWS * 1e9 for t in interp_times]

    print(f"[item3] codegen    ns/row: best={min(codegen_ns_row):.2f} range={min(codegen_ns_row):.2f}-{max(codegen_ns_row):.2f}")
    print(f"[item3] interpreter ns/row: best={min(interp_ns_row):.2f} range={min(interp_ns_row):.2f}-{max(interp_ns_row):.2f}")
    print(f"[item3] ratio (interp/codegen), best-of: {min(interp_ns_row)/min(codegen_ns_row):.2f}x")

    append_result({
        "item": "compare_vs_codegen", "n_rows": N_ROWS, "reps": REPS,
        "exact_equal": bool(exact_equal),
        "codegen_ns_row_all": codegen_ns_row, "interp_ns_row_all": interp_ns_row,
        "codegen_ns_row_best": min(codegen_ns_row), "interp_ns_row_best": min(interp_ns_row),
        "n_segments": len(driver.segments), "segment_kind": seg.kind,
        "kernel_fn_type": str(type(seg.kernel_fn)),
    })

    # --- item 6: retune, both engines, signature counts must not grow ---
    codegen_sig_before = len(driver.signatures)
    interp_sig_before = len(run_program.signatures)

    retuned_scalar = dict(default_scalar)
    retuned_scalar[(step_name, "term_still_short__floor")] = 25.0
    resolved_retuned = ResolvedParams(per_step_scalar=retuned_scalar, per_step_bundle={}, shared=None)

    registry_r = {"high_income": high_income.copy(), "term_cap": term_cap0.copy()}
    seg.run(registry_r, resolved_retuned, N_ROWS)
    out_codegen_retuned = registry_r["term_cap"]

    tmpl_retuned = param_template({4: 25.0})  # REG_FLOOR = 4
    out_interp_retuned = np.empty(N_ROWS, dtype=np.float64)
    run_program(prog.op, prog.step_idx, prog.arg0, prog.arg1, prog.arg2, prog.dest,
                prog.next_, prog.alt, prog.start_pc, PTR_TABLE, tmpl_retuned, row_regs, N_ROWS, out_interp_retuned)

    codegen_sig_after = len(driver.signatures)
    interp_sig_after = len(run_program.signatures)

    codegen_changed = not np.array_equal(out_codegen, out_codegen_retuned)
    interp_changed = not np.array_equal(out_interp, out_interp_retuned)
    retuned_agree = np.array_equal(out_codegen_retuned, out_interp_retuned)

    print(f"[item6] codegen: retuned output differs from default: {codegen_changed}; "
          f"signatures {codegen_sig_before} -> {codegen_sig_after}")
    print(f"[item6] interpreter: retuned output differs from default: {interp_changed}; "
          f"run_program.signatures {interp_sig_before} -> {interp_sig_after}")
    print(f"[item6] codegen and interpreter still agree after retune: {retuned_agree}")

    assert codegen_sig_before == codegen_sig_after, "codegen retune grew the signature count -- it recompiled"
    assert interp_sig_before == interp_sig_after, "interpreter retune grew the signature count -- it recompiled"
    assert codegen_changed and interp_changed, "retune had no effect -- test is not exercising anything"
    assert retuned_agree, "codegen and interpreter disagree after an identical retune"

    append_result({
        "item": "params_retune",
        "codegen_signatures_before": codegen_sig_before, "codegen_signatures_after": codegen_sig_after,
        "interp_signatures_before": interp_sig_before, "interp_signatures_after": interp_sig_after,
        "codegen_output_changed": bool(codegen_changed), "interp_output_changed": bool(interp_changed),
        "retuned_outputs_agree": bool(retuned_agree),
    })

    print("\nAll assertions passed: codegen and interpreter agree exactly, before and after a retune;")
    print("neither engine's compiled-signature count grew on retune.")


if __name__ == "__main__":
    main()
