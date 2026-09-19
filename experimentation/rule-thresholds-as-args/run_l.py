"""EXPERIMENT L -- rule thresholds as arguments vs emitted literals.

Tests doc 05 S4.2's untested assertion:
  "No decision-relevant constant is emitted into driver source. Params are
   already arguments so that path is safe. A ruleset's thresholds are NOT
   yet -- and that is precisely the path a business-user UI edits."

Connects EXPERIMENTS.md SS C (stale-cache-on-constant-edit), G (disabled
rules still cost full compile time; compile ~ lines^1.4), and doc 08 S2's
three change classes (values / interiors / skeleton).

Five parts, run in order, each writing to results.jsonl immediately:
  2. compile time,  literal vs args, 3/10/30 rules
  1. runtime cost of the indirection, 100k/1M rows, 10/30 rules
  3. does changing a threshold recompile? N retunes, both forms
  3b. stale-cache hazard: reproduced in literal form, shown impossible in args form
  4. enablement-as-argument (mask): runtime cost vs compile cost avoided

Reused, not rewritten:
  - rss_mb / med / pct / compile_kernel-style exec+njit+call-once / count_compiles
    (numba.core.event.install_recorder pattern) -- imported LIVE from
    experimentation/staged-compile-atomic-swap/run.py (module `swaprun` below)
  - elapsed()/log() progress-clock pattern -- experimentation/prange-crossover/
  - median-of-repeats timing (warm call, then time `repeats` calls, report median)
    -- experimentation/dtype-boundary/dtype_boundary.py:median_us /
       experimentation/output-writeback-convention/writeback.py:median_s
  - the same-byte-length constant-edit trick for the stale-cache repro
    -- experimentation/numba_cache_survival/run_experiment.py:part_stale
  - write generated source to a REAL .py file, import via
    importlib.util.spec_from_file_location (never exec()) for the stale-cache
    part specifically (compile-time/runtime parts use exec() like
    staged-compile-atomic-swap/run.py:compile_kernel does, since no cache
    persistence is being tested there) -- doc 05 S4.1 convention, reused from
    experimentation/ruleset-compile-latency/run.py:load_source's pattern

MEMORY: every array here is at most 1,000,000 rows x 16 f8 feature columns +
1 i8 output column (~136 MB) -- see the printed estimate below, well under
the 4GB restructure threshold, so this experiment runs IN-PROCESS, not under
tmux/systemd-run (that machinery is for runs expected to cross ~2GB; using it
here would only add overhead for no benefit). `free -g` is checked and must
show >=8GB available before anything runs. Results are still written
incrementally to results.jsonl, flushed+fsync'd after every record, per the
mandatory rules -- a kill costs one data point, not the run.

Run:
    /path/to/.venv/bin/python run_l.py
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import numba
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emit_l as E  # noqa: E402

# reused live, not copy-pasted -- rss_mb, med, pct, count_compiles
sys.path.insert(0, str(HERE.parent / "staged-compile-atomic-swap"))
import run as swaprun  # noqa: E402

T0 = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


RESULTS_PATH = HERE / "results.jsonl"
PEAK_RSS_MB = [0.0]


def emit_record(rec: dict) -> None:
    rec["t"] = round(elapsed(), 3)
    r = swaprun.rss_mb()
    PEAK_RSS_MB[0] = max(PEAK_RSS_MB[0], r)
    rec["rss_mb"] = round(r, 1)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_memory_gate(min_available_gb: int = 8) -> int:
    out = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("Mem:"))
    parts = line.split()
    available = int(parts[6]) if len(parts) > 6 else int(parts[3])
    log(f"free -g -> {line.strip()}  (available={available}G)")
    if available < min_available_gb:
        raise SystemExit(f"REFUSING TO START: available {available}G < {min_available_gb}G floor (mandatory rule 5)")
    return available


def median_s(fn, repeats: int):
    fn()  # warm
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


def estimate_bytes(rows: int) -> int:
    """N_FLOAT f8 feature columns + 1 i8 output column."""
    return rows * (E.N_FLOAT * 8 + 8)


def make_feature_args(rows: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    fs = tuple(np.ascontiguousarray(rng.random(rows)) for _ in range(E.N_FLOAT))
    out = np.empty(rows, dtype=np.int64)
    return fs, out


def compile_literal(rules, fname: str):
    src = E.emit_literal(rules, fname=fname)
    ns: dict = {}
    exec(compile(src, f"<{fname}>", "exec"), ns)
    disp = njit(cache=False, nogil=True)(ns[fname])
    return disp, src


def compile_args(rules, n_clauses: int, fname: str):
    src = E.emit_args(rules, n_clauses, fname=fname)
    ns: dict = {}
    exec(compile(src, f"<{fname}>", "exec"), ns)
    disp = njit(cache=False, nogil=True)(ns[fname])
    return disp, src


def compile_args_masked(rules, n_clauses: int, fname: str):
    src = E.emit_args_masked(rules, n_clauses, fname=fname)
    ns: dict = {}
    exec(compile(src, f"<{fname}>", "exec"), ns)
    disp = njit(cache=False, nogil=True)(ns[fname])
    return disp, src


# --------------------------------------------------------------------------- PART 2
def part2_compile_time(n_clauses: int = 3, seed: int = 0, warm_rows: int = 1000):
    log("=== PART 2: compile time, literal vs args -- 3/10/30 rules ===")
    for n_rules in (3, 10, 30):
        rules = E.gen_rule_spec(n_rules, n_clauses=n_clauses, seed=seed)
        fs, out = make_feature_args(warm_rows)
        th = E.thresholds_array(rules, n_clauses)

        t0 = time.perf_counter()
        disp_lit, src_lit = compile_literal(rules, fname=f"lit{n_rules}")
        disp_lit(*fs, out)
        c_lit = time.perf_counter() - t0
        lines_lit = E.count_lines(src_lit)

        t0 = time.perf_counter()
        disp_arg, src_arg = compile_args(rules, n_clauses, fname=f"arg{n_rules}")
        disp_arg(*fs, out, th)
        c_arg = time.perf_counter() - t0
        lines_arg = E.count_lines(src_arg)

        log(f"n_rules={n_rules:3d}  literal: {lines_lit:4d} lines {c_lit:6.3f}s   "
            f"args: {lines_arg:4d} lines {c_arg:6.3f}s   "
            f"line_delta={lines_lit - lines_arg:+d}  compile_time_delta={c_lit - c_arg:+.3f}s")
        emit_record({"phase": "compile_time", "n_rules": n_rules,
                     "literal_lines": lines_lit, "literal_compile_s": c_lit,
                     "args_lines": lines_arg, "args_compile_s": c_arg})
        del disp_lit, disp_arg
    gc.collect()


# --------------------------------------------------------------------------- PART 1
def part1_runtime_cost(n_clauses: int = 3, seed: int = 0, repeats: int = 41):
    log("=== PART 1: runtime cost of the indirection -- 100k/1M rows, 10/30 rules ===")
    for n_rules in (10, 30):
        rules = E.gen_rule_spec(n_rules, n_clauses=n_clauses, seed=seed)
        th = E.thresholds_array(rules, n_clauses)
        warm_fs, warm_out = make_feature_args(1000)
        disp_lit, _ = compile_literal(rules, fname=f"litrt{n_rules}")
        disp_lit(*warm_fs, warm_out)
        disp_arg, _ = compile_args(rules, n_clauses, fname=f"argrt{n_rules}")
        disp_arg(*warm_fs, warm_out, th)

        for rows in (100_000, 1_000_000):
            est = estimate_bytes(rows)
            log(f"  n_rules={n_rules} rows={rows:,}: estimated feature+out bytes = "
                f"{est / 1e6:.1f} MB (mandatory rule 1 estimate, before running)")
            fs, out = make_feature_args(rows)

            t_lit, s_lit = median_s(lambda: disp_lit(*fs, out), repeats)
            t_arg, s_arg = median_s(lambda: disp_arg(*fs, out, th), repeats)
            ns_lit = t_lit * 1e9 / rows
            ns_arg = t_arg * 1e9 / rows
            iqr_lit_ns = (statistics.quantiles(s_lit, n=4)[2] - statistics.quantiles(s_lit, n=4)[0]) * 1e9 / rows
            iqr_arg_ns = (statistics.quantiles(s_arg, n=4)[2] - statistics.quantiles(s_arg, n=4)[0]) * 1e9 / rows
            pct_slower = 100.0 * (t_arg - t_lit) / t_lit
            log(f"    literal {ns_lit:6.2f} ns/row (IQR {iqr_lit_ns:.2f})   "
                f"args {ns_arg:6.2f} ns/row (IQR {iqr_arg_ns:.2f})   "
                f"args is {pct_slower:+.1f}% vs literal   n={repeats}")
            emit_record({"phase": "runtime_cost", "n_rules": n_rules, "rows": rows, "repeats": repeats,
                         "literal_ns_per_row": ns_lit, "args_ns_per_row": ns_arg,
                         "literal_iqr_ns_per_row": iqr_lit_ns, "args_iqr_ns_per_row": iqr_arg_ns,
                         "pct_slower_args_vs_literal": pct_slower})
            del fs, out
            gc.collect()
        del disp_lit, disp_arg
    gc.collect()


# --------------------------------------------------------------------------- PART 3
def part3_retune(n_rules: int = 5, n_clauses: int = 3, seed: int = 0,
                  n_retunes: int = 8, rows: int = 20_000):
    log(f"=== PART 3: {n_retunes} retunes -- does changing a threshold recompile? ===")
    rules = E.gen_rule_spec(n_rules, n_clauses=n_clauses, seed=seed)
    fs, out = make_feature_args(rows)
    rng = random.Random(seed ^ 0xF00D)

    # ARGS form: ONE compile, then N retunes as pure data swaps.
    disp_arg, _ = compile_args(rules, n_clauses, fname="argretune")
    th0 = E.thresholds_array(rules, n_clauses)
    disp_arg(*fs, out, th0)
    sig_before = len(disp_arg.signatures)
    events_args = 0
    for _ in range(n_retunes):
        th_k = th0 + rng.uniform(-0.05, 0.05)
        _, n_events = swaprun.count_compiles(lambda: disp_arg(*fs, out, th_k))
        events_args += (n_events or 0)
    sig_after = len(disp_arg.signatures)
    log(f"  args form:    driver.signatures {sig_before} -> {sig_after} across {n_retunes} retunes, "
        f"compile events = {events_args}")
    emit_record({"phase": "retune", "form": "args", "n_retunes": n_retunes,
                 "signatures_before": sig_before, "signatures_after": sig_after,
                 "compile_events": events_args})

    # LITERAL form: each retune = new source text = new dispatcher = a real recompile.
    events_lit = 0
    total_compile_s = 0.0
    for k in range(n_retunes):
        rules_k = [[(feat, op, round(thresh + rng.uniform(-0.05, 0.05), 4)) for feat, op, thresh in clauses]
                   for clauses in rules]
        t0 = time.perf_counter()
        disp_k, _ = compile_literal(rules_k, fname=f"litretune{k}")
        disp_k(*fs, out)
        total_compile_s += time.perf_counter() - t0
        events_lit += 1
        del disp_k
    log(f"  literal form: {events_lit} full recompiles across {n_retunes} retunes, "
        f"{total_compile_s:.2f}s total ({total_compile_s / n_retunes * 1000:.0f} ms/retune avg)")
    emit_record({"phase": "retune", "form": "literal", "n_retunes": n_retunes,
                 "recompiles": events_lit, "total_compile_s": total_compile_s})
    del disp_arg, fs, out
    gc.collect()


# -------------------------------------------------------------------------- PART 3b
def part3b_stale_cache(n_clauses: int = 3, seed: int = 0, rows: int = 5000):
    log("=== PART 3b: stale-cache hazard -- reproduced (literal) vs impossible (args) ===")
    work = HERE / "_stale_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    rules = E.gen_rule_spec(3, n_clauses=n_clauses, seed=seed)
    fs, _ = make_feature_args(rows)

    # ---- literal form: real .py file, @njit(cache=True), default caching
    # (CPython .pyc + numba .nbi/.nbc both left on, exactly as EXPERIMENTS.md
    # SS C's original harness ran -- this is "the hazard" as documented). ----
    lit_body = E.emit_literal(rules, fname="drv")
    lit_src = "from numba import njit\n\n\n@njit(cache=True)\n" + lit_body
    m = re.search(r"0\.\d{4}", lit_src)
    assert m, "no literal threshold token found to edit"
    old_tok = m.group(0)
    new_tok = "0." + "".join(str((int(c) + 5) % 10) for c in old_tok[2:])
    assert len(new_tok) == len(old_tok)
    old_line = next(ln for ln in lit_src.splitlines() if old_tok in ln)
    new_line = old_line.replace(old_tok, new_tok, 1)

    pm = work / "drv_lit.py"
    pm.write_text(lit_src)

    def load_and_run(path: Path, modname: str):
        spec = importlib.util.spec_from_file_location(modname, str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = np.empty(rows, dtype=np.int64)
        mod.drv(*fs, out)
        return out

    before_edit = load_and_run(pm, "stale_lit_mod_a")

    st = os.stat(pm)
    keep = (st.st_atime, st.st_mtime)
    body = pm.read_text()
    assert old_line in body
    pm.write_text(body.replace(old_line, new_line, 1))
    os.utime(pm, keep)
    assert os.stat(pm).st_size == st.st_size, "edit changed file size -- trick invalid"

    # (a) as-deployed: both CPython .pyc and numba .nbi/.nbc left untouched.
    after_edit_untouched = load_and_run(pm, "stale_lit_mod_a")  # SAME modname -> same .pyc key
    stale_asdeployed = bool(np.array_equal(before_edit, after_edit_untouched))

    # (b) clear ONLY CPython's .pyc, leave numba's own .nbi/.nbc in place --
    # isolates whether numba's OWN cache (not CPython's) is independently stale.
    pycache = work / "__pycache__"
    if pycache.exists():
        for pyc in pycache.glob("*.pyc"):
            pyc.unlink()
    after_edit_pyc_cleared = load_and_run(pm, "stale_lit_mod_b")  # fresh modname forces re-parse
    stale_numba_only = bool(np.array_equal(before_edit, after_edit_pyc_cleared))

    # (c) TRUE ground truth: compute the edited rule set's answer via a
    # completely independent, never-cached path (fresh exec + cache=False),
    # so this cannot be stale by construction -- this is what "edit actually
    # changes ground truth" must be checked against, not (b).
    rules_edited = [list(clauses) for clauses in rules]
    rules_edited[0] = list(rules_edited[0])
    feat0, op0, _old_th = rules_edited[0][0]
    new_th_val = float(new_tok)
    rules_edited[0][0] = (feat0, op0, new_th_val)
    disp_truth, _ = compile_literal(rules_edited, fname="drv_truth_nocache")
    out_truth = np.empty(rows, dtype=np.int64)
    disp_truth(*fs, out_truth)
    edit_matters = not bool(np.array_equal(out_truth, before_edit))
    numba_cache_still_stale_after_pyc_clear = bool(
        np.array_equal(after_edit_pyc_cleared, before_edit) and not np.array_equal(out_truth, before_edit)
    )

    log(f"  LITERAL (a) as-deployed (.pyc+.nbi/.nbc both stale-eligible): served pre-edit value = {stale_asdeployed}")
    log(f"  LITERAL (b) CPython .pyc cleared, numba .nbi/.nbc left alone: served pre-edit value = {stale_numba_only}")
    log(f"  LITERAL (c) true ground truth (fresh cache=False compile):    edit actually changes output = {edit_matters}")
    if numba_cache_still_stale_after_pyc_clear:
        log("  -> numba's OWN cache is independently stale here, not just CPython's .pyc "
            "(refines EXPERIMENTS.md K finding #2, which could not construct this case)")
    emit_record({"phase": "stale_cache", "form": "literal",
                 "stale_as_deployed": stale_asdeployed,
                 "stale_with_pyc_cleared_numba_cache_only": stale_numba_only,
                 "edit_changes_true_ground_truth": edit_matters,
                 "numba_cache_independently_stale": numba_cache_still_stale_after_pyc_clear,
                 "old_token": old_tok, "new_token": new_tok})
    del disp_truth
    stale = stale_asdeployed

    # ---- args form: there is no threshold text in the file to edit. A
    # "retune" is a new th array passed to the SAME dispatcher; the file's
    # bytes, mtime and size never move, so there is no (mtime,size) cache key
    # for a stale edit to collide on -- not "not yet observed", structurally
    # absent. ----
    arg_src = E.emit_args(rules, n_clauses, fname="drv")
    pa = work / "drv_arg.py"
    pa.write_text(arg_src)
    spec = importlib.util.spec_from_file_location("stale_arg_mod", str(pa))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    disp_arg = njit(cache=True, nogil=True)(mod.drv)

    th_a = E.thresholds_array(rules, n_clauses)
    out1 = np.empty(rows, dtype=np.int64)
    disp_arg(*fs, out1, th_a)
    st_before = os.stat(pa)

    th_b = th_a.copy()
    th_b[0] = round((th_b[0] + 0.5) % 1.0, 4)
    out2 = np.empty(rows, dtype=np.int64)
    disp_arg(*fs, out2, th_b)  # the "retune": zero file writes
    st_after = os.stat(pa)

    file_untouched = (st_before.st_mtime == st_after.st_mtime and st_before.st_size == st_after.st_size)
    output_changed_correctly = not bool(np.array_equal(out1, out2))
    log(f"  ARGS:    retuned by passing a new th array. file (mtime,size) unchanged = {file_untouched}, "
        f"output correctly changed = {output_changed_correctly}, driver.signatures = {len(disp_arg.signatures)}")
    emit_record({"phase": "stale_cache", "form": "args", "file_untouched": file_untouched,
                 "output_correctly_changed": output_changed_correctly, "signatures": len(disp_arg.signatures)})

    shutil.rmtree(work, ignore_errors=True)
    gc.collect()


# --------------------------------------------------------------------------- PART 4
def part4_mask_breakeven(n_clauses: int = 3, seed: int = 0, rows: int = 500_000, repeats: int = 41):
    log("=== PART 4: enablement as an argument (mask) -- overhead vs compile cost avoided ===")
    for n_rules in (10, 30):
        rules = E.gen_rule_spec(n_rules, n_clauses=n_clauses, seed=seed)
        th = E.thresholds_array(rules, n_clauses)
        mask_on = np.ones(n_rules, dtype=np.bool_)
        warm_fs, warm_out = make_feature_args(1000)

        disp_arg, _ = compile_args(rules, n_clauses, fname=f"argmaskbase{n_rules}")
        disp_arg(*warm_fs, warm_out, th)
        disp_mask, _ = compile_args_masked(rules, n_clauses, fname=f"argmaskm{n_rules}")
        disp_mask(*warm_fs, warm_out, th, mask_on)

        t0 = time.perf_counter()
        disp_lit_fresh, _ = compile_literal(rules, fname=f"argmasklit{n_rules}")
        disp_lit_fresh(*warm_fs, warm_out)
        recompile_cost_s = time.perf_counter() - t0

        fs, out = make_feature_args(rows)
        t_arg, _ = median_s(lambda: disp_arg(*fs, out, th), repeats)
        t_mask, _ = median_s(lambda: disp_mask(*fs, out, th, mask_on), repeats)
        overhead_ns_row = (t_mask - t_arg) * 1e9 / rows
        overhead_pct = 100.0 * (t_mask - t_arg) / t_arg
        breakeven_rows = (recompile_cost_s / (overhead_ns_row * 1e-9)) if overhead_ns_row > 0 else float("inf")

        log(f"  n_rules={n_rules}: mask overhead {overhead_pct:+.2f}% ({overhead_ns_row:+.4f} ns/row), "
            f"literal recompile {recompile_cost_s:.2f}s -> break-even {breakeven_rows:,.0f} rows "
            f"scored on the mask path per toggle to match one recompile's cost")
        emit_record({"phase": "mask_breakeven", "n_rules": n_rules, "rows": rows,
                     "args_ms": t_arg * 1e3, "args_mask_ms": t_mask * 1e3,
                     "overhead_pct": overhead_pct, "overhead_ns_per_row": overhead_ns_row,
                     "literal_recompile_s": recompile_cost_s, "breakeven_rows": breakeven_rows})
        del disp_arg, disp_mask, disp_lit_fresh, fs, out
        gc.collect()


def main():
    check_memory_gate(min_available_gb=8)
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
    log(f"env: python numba={numba.__version__} numpy={np.__version__}")
    est_1m = estimate_bytes(1_000_000)
    log(f"ESTIMATE before largest configuration (1,000,000 rows x {E.N_FLOAT} f8 cols + 1 i8 out): "
        f"{est_1m / 1e6:.1f} MB -- under the 4GB restructure threshold and under the 2GB "
        f"tmux/systemd-run threshold, so this runs in-process (stated explicitly, not skipped silently)")

    part2_compile_time()
    part1_runtime_cost()
    part3_retune()
    part3b_stale_cache()
    part4_mask_breakeven()

    log(f"peak RSS observed this process: {PEAK_RSS_MB[0]:.1f} MB")
    log("DONE")


if __name__ == "__main__":
    main()
