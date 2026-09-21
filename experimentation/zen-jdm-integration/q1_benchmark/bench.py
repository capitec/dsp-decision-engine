"""Q1 -- could ZEN be decider2's execution engine, and what would it cost?

Measures, on the IDENTICAL rule set (rules.py -> build_decider2.py /
build_zen.py), IDENTICAL data, with an identical-answers assertion before
any timing is trusted:

  1. cold start: config arrives -> first answer (compile/parse cost)
  2. single-record latency: decider2 pipeline.score() vs zen decision.evaluate()
  3. batch throughput at 100k rows: decider2 apply() (fused, njit) vs
     ZEN's two available batch shapes -- a manual per-row Python loop
     calling decision.evaluate() (the shape a Polars map_elements call has,
     per the research note's S8.4), and ZenEngine.evaluate_batch() with a
     static loader and one request per row (the best-case batch call the
     Python binding exposes).

Memory discipline: 100k rows of a handful of f8/i8 columns is a few MB.
Writes each measurement to results.jsonl immediately (append, flush) so a
crash mid-run does not lose prior numbers.
"""
from __future__ import annotations

import gc
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_decider2  # noqa: E402
import build_zen  # noqa: E402
from rules import oracle  # noqa: E402

import zen  # noqa: E402

RESULTS = HERE / "results.jsonl"
N_BATCH = 100_000
N_SINGLE = 5_000
SEED = 20260921


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def record(**kw) -> None:
    kw["ts"] = time.time()
    with RESULTS.open("a") as f:
        f.write(json.dumps(kw, default=str) + "\n")
    log(f"recorded: {kw}")


def percentiles(samples_s: list[float]) -> dict:
    xs = sorted(samples_s)
    n = len(xs)

    def pct(p):
        return xs[min(n - 1, int(n * p))]

    return {
        "n": n,
        "min_ns": xs[0] * 1e9,
        "p50_ns": statistics.median(xs) * 1e9,
        "p95_ns": pct(0.95) * 1e9,
        "p99_ns": pct(0.99) * 1e9,
        "max_ns": xs[-1] * 1e9,
    }


def make_incomes(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mixed: uniform over a wide range, with mass at/near band boundaries
    # (the case both engines' comparison operators have to get exactly
    # right) and some far outside [0, 400000] to exercise the open edges.
    base = rng.uniform(-10_000, 600_000, size=n)
    boundary_mask = rng.random(n) < 0.1
    boundaries = np.array([b[0] for b in __import__("rules").BANDS if b[0] is not None], dtype=np.float64)
    base[boundary_mask] = rng.choice(boundaries, size=boundary_mask.sum())
    return base


# ---------------------------------------------------------------------------
# 0. Identical-answers assertion (must pass before any timing counts)
# ---------------------------------------------------------------------------

def assert_identical(pipeline, shared, decision, incomes: np.ndarray) -> None:
    log(f"asserting identical answers on {len(incomes)} incomes (oracle + decider2 + zen)...")
    mismatches = []
    for inc in incomes:
        inc = float(inc)
        exp_code, exp_limit = oracle(inc)
        d2 = pipeline.score({"income": inc}, shared=shared)
        z = decision.evaluate({"income": inc})["result"]
        z_code, z_limit = int(z["tier_code"]), float(z["limit"])
        if not (d2["tier_code"] == exp_code == z_code and d2["limit"] == exp_limit == z_limit):
            mismatches.append((inc, exp_code, exp_limit, d2["tier_code"], d2["limit"], z_code, z_limit))
    if mismatches:
        for m in mismatches[:20]:
            log(f"  MISMATCH income={m[0]} oracle=({m[1]},{m[2]}) decider2=({m[3]},{m[4]}) zen=({m[5]},{m[6]})")
        raise AssertionError(f"{len(mismatches)}/{len(incomes)} mismatches — see log above")
    log(f"OK: all {len(incomes)} answers identical across oracle, decider2 and ZEN.")
    record(phase="identical_answers", n=len(incomes), result="PASS")


# ---------------------------------------------------------------------------
# 1. Cold start
# ---------------------------------------------------------------------------

def measure_cold_start() -> None:
    log("=== cold start: config arrives -> first answer ===")

    # decider2: DecisionTable() construction -> table_module() (numba
    # compile, cold cache dir) -> first score() call.
    with tempfile.TemporaryDirectory() as bd:
        t0 = time.perf_counter()
        table = build_decider2.build_table()
        t_doc = time.perf_counter()
        from decider2 import flow
        from decider2.tables import table_module
        tm = table_module(table, build_dir=bd)
        t_compiled = time.perf_counter()
        pipeline = flow(tm.module)
        first = pipeline.score({"income": 42000.0}, shared=tm.shared)
        t_first = time.perf_counter()
        record(
            phase="cold_start", engine="decider2",
            doc_build_s=t_doc - t0,
            compile_s=t_compiled - t_doc,
            first_call_s=t_first - t_compiled,
            total_s=t_first - t0,
            first_answer=first,
        )

    # ZEN: JSON content string arrives -> create_decision (parse/validate)
    # -> first evaluate() call. No AOT compile step exists on this path —
    # that asymmetry (decider2 pays a compile, ZEN does not) is the point
    # of measuring both.
    t0 = time.perf_counter()
    content = build_zen.build_content()
    content_str = json.dumps(content)
    t_doc = time.perf_counter()
    engine = zen.ZenEngine()
    decision = engine.create_decision(content_str)
    t_created = time.perf_counter()
    first = decision.evaluate({"income": 42000.0})["result"]
    t_first = time.perf_counter()
    record(
        phase="cold_start", engine="zen",
        doc_build_s=t_doc - t0,
        compile_s=t_created - t_doc,
        first_call_s=t_first - t_created,
        total_s=t_first - t0,
        first_answer=first,
    )

    # Second process-level detail worth recording separately: importing
    # `zen` itself and `numba`/decider2's compile machinery are both
    # one-time process costs paid before any of the above. Both are
    # already-paid by the time this function runs (imported at module
    # load), so this function measures the DOCUMENT'S cost only, matching
    # "config arrives" — not interpreter/process startup.


# ---------------------------------------------------------------------------
# 2. Single-record latency
# ---------------------------------------------------------------------------

def measure_single_record(pipeline, shared, decision, incomes: np.ndarray) -> None:
    log(f"=== single-record latency: n={N_SINGLE} ===")

    # decider2 .score()
    samples = []
    gc.collect()
    for inc in incomes[:N_SINGLE]:
        rec = {"income": float(inc)}
        t0 = time.perf_counter()
        pipeline.score(rec, shared=shared)
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    stats = percentiles(samples)
    record(phase="single_record", engine="decider2", method="score()", **stats)

    # ZEN ZenDecision.evaluate() — the direct call, no batching, no loader
    # indirection: this is the "realtime path" analogue to decider2.score().
    samples = []
    gc.collect()
    for inc in incomes[:N_SINGLE]:
        ctx = {"income": float(inc)}
        t0 = time.perf_counter()
        decision.evaluate(ctx)
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    stats = percentiles(samples)
    record(phase="single_record", engine="zen", method="ZenDecision.evaluate()", **stats)

    # ZEN with trace=False explicit and options reused (options dict built
    # once, not per call) — checks whether the TypedDict options arg itself
    # is a measurable per-call cost.
    samples = []
    gc.collect()
    opts = {"trace": False}
    for inc in incomes[:N_SINGLE]:
        ctx = {"income": float(inc)}
        t0 = time.perf_counter()
        decision.evaluate(ctx, opts)
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    stats = percentiles(samples)
    record(phase="single_record", engine="zen", method="ZenDecision.evaluate(ctx, {trace:False})", **stats)


# ---------------------------------------------------------------------------
# 3. Batch throughput at 100k rows
# ---------------------------------------------------------------------------

def measure_batch(pipeline, shared, decision, content_str: str, incomes: np.ndarray) -> None:
    n = len(incomes)
    log(f"=== batch throughput: n={n} ===")

    # decider2: apply(), fused njit driver, polars in/out.
    frame = pl.DataFrame({"income": incomes})
    # warm the dispatcher once (first call may include dispatch-site setup
    # separate from the numba compile already paid in build_decider2.build)
    pipeline.apply(frame.head(10), shared=shared)
    gc.collect()
    t0 = time.perf_counter()
    out = pipeline.apply(frame, shared=shared, mode="fused")
    t1 = time.perf_counter()
    d2_elapsed = t1 - t0
    record(
        phase="batch", engine="decider2", method="apply(mode='fused')",
        n=n, elapsed_s=d2_elapsed, ns_per_row=d2_elapsed / n * 1e9,
        rows_per_s=n / d2_elapsed,
    )
    d2_tier = out["tier_code"].to_numpy()
    d2_limit = out["limit"].to_numpy()

    # ZEN, shape A: manual per-row Python loop calling ZenDecision.evaluate()
    # — the shape a Polars map_elements call has (research note S8.4).
    gc.collect()
    zen_codes = np.empty(n, dtype=np.int64)
    zen_limits = np.empty(n, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(n):
        r = decision.evaluate({"income": float(incomes[i])})["result"]
        zen_codes[i] = int(r["tier_code"])
        zen_limits[i] = float(r["limit"])
    t1 = time.perf_counter()
    zen_loop_elapsed = t1 - t0
    record(
        phase="batch", engine="zen", method="per-row Python loop, ZenDecision.evaluate()",
        n=n, elapsed_s=zen_loop_elapsed, ns_per_row=zen_loop_elapsed / n * 1e9,
        rows_per_s=n / zen_loop_elapsed,
    )

    # ZEN, shape B: ZenEngine.evaluate_batch() with a static loader, one
    # request per row, same decision reused via its key — the best-case
    # batch call this Python binding exposes (found empirically; not in
    # the docs page cited by the research note).
    content = json.loads(content_str)
    batch_engine = zen.ZenEngine({"loader": {"type": "static", "content": {"rules.json": content}}})
    requests = [{"key": "rules.json", "context": {"income": float(x)}} for x in incomes]
    gc.collect()
    t0 = time.perf_counter()
    results = batch_engine.evaluate_batch(requests)
    t1 = time.perf_counter()
    zen_batch_elapsed = t1 - t0
    record(
        phase="batch", engine="zen", method="ZenEngine.evaluate_batch(), static loader",
        n=n, elapsed_s=zen_batch_elapsed, ns_per_row=zen_batch_elapsed / n * 1e9,
        rows_per_s=n / zen_batch_elapsed,
    )
    zenb_codes = np.array([int(r["data"]["result"]["tier_code"]) for r in results], dtype=np.int64)
    zenb_limits = np.array([float(r["data"]["result"]["limit"]) for r in results], dtype=np.float64)

    # Identical answers at scale, all three: decider2 apply(), zen loop, zen batch.
    assert np.array_equal(d2_tier, zen_codes), "decider2 vs zen-loop tier_code mismatch at scale"
    assert np.allclose(d2_limit, zen_limits), "decider2 vs zen-loop limit mismatch at scale"
    assert np.array_equal(zen_codes, zenb_codes), "zen-loop vs zen-batch tier_code mismatch"
    assert np.allclose(zen_limits, zenb_limits), "zen-loop vs zen-batch limit mismatch"
    log(f"OK: all {n} batch answers identical across decider2, zen-loop, zen-batch.")
    record(phase="identical_answers_batch", n=n, result="PASS")

    record(
        phase="batch_summary",
        decider2_ns_per_row=d2_elapsed / n * 1e9,
        zen_loop_ns_per_row=zen_loop_elapsed / n * 1e9,
        zen_batch_ns_per_row=zen_batch_elapsed / n * 1e9,
        zen_loop_vs_decider2_x=zen_loop_elapsed / d2_elapsed,
        zen_batch_vs_decider2_x=zen_batch_elapsed / d2_elapsed,
    )


def main() -> None:
    RESULTS.unlink(missing_ok=True)
    record(phase="meta", n_batch=N_BATCH, n_single=N_SINGLE, seed=SEED)

    measure_cold_start()

    with tempfile.TemporaryDirectory() as bd:
        log("building warm decider2 pipeline (separate cache dir from cold-start measurement)...")
        pipeline, shared = build_decider2.build(bd)
        content = build_zen.build_content()
        content_str = json.dumps(content)
        engine = zen.ZenEngine()
        decision = engine.create_decision(content_str)

        incomes = make_incomes(N_BATCH, SEED)
        assert_identical(pipeline, shared, decision, incomes[:2000])
        measure_single_record(pipeline, shared, decision, incomes)
        measure_batch(pipeline, shared, decision, content_str, incomes)

    log("DONE — wrote results.jsonl")


if __name__ == "__main__":
    main()
