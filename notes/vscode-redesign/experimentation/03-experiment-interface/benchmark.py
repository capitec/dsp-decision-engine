"""Local-scale benchmark for the experiment spike: how many rows can one local
run and one scenario sweep realistically carry?

Run::

    PYTHONPATH=tools/decider-bridge uv run python \
        notes/vscode-redesign/experimentation/03-experiment-interface/benchmark.py
"""
from __future__ import annotations

import time

import polars as pl

from decider import Engine, flow, param
import decider_bridge.forks as forks


def ratio(income: float, debt: float) -> float:
    return debt / income if income > 0 else 1.0


def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


def tier(approved: bool, income: float) -> float:
    return 1.0 if approved else (0.5 if income > 0 else 0.0)


pipeline = flow(ratio, approved, tier, name="demo")


def bench_run(mode, n, reps=3):
    df = pl.DataFrame({"income": [1000.0] * n, "debt": [float(i) for i in range(n)]})
    exe = Engine().bind(pipeline, mode=mode)
    exe.run(df)  # warm the kernel / cache
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        exe.run(df)
        best = min(best, time.perf_counter() - t)
    return best


def bench_sweep(n, scenarios=5, reps=3):
    """The experiment path: forks.fork is interpreted, replaying a fresh session per scenario."""
    df = pl.DataFrame({"income": [1000.0] * n, "debt": [float(i) for i in range(n)]})
    params = pipeline.parameters().defaults()
    scs = [{"name": f"s{i}", "params": {"demo": {"approved": {"limit": 0.1 + i * 0.1}}}} for i in range(scenarios)]
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        forks.sweep(pipeline, df, params, [], None, scs)
        best = min(best, time.perf_counter() - t)
    return best


def bench_affordability_score(reps=2000):
    """Real pipeline (02-affordability): per-record score cost, interpreted."""
    import datetime
    import json
    from pathlib import Path

    import pipeline as aff
    built = aff.build()
    params = built.parameters().defaults()
    req = json.loads(Path("example_projects/02-affordability/sonnet/sample_request.json").read_text())
    req["decision_date"] = datetime.date.fromisoformat(req["decision_date"])
    req["bureau_as_of_date"] = datetime.date.fromisoformat(req["bureau_as_of_date"])
    exe = Engine().bind(built, mode="interpreted")
    t = time.perf_counter()
    for _ in range(reps):
        exe.score(req, params=params)
    return (time.perf_counter() - t) / reps


if __name__ == "__main__":
    sizes = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]
    print("== run() throughput, one frame (best of 3) ==")
    print(f"{'rows':>12} | {'fused':>12} | {'stepped':>12} | {'interpreted':>12}")
    for n in sizes:
        row = []
        for mode in ("fused", "stepped", "interpreted"):
            try:
                t = bench_run(mode, n)
                row.append(f"{t * 1000:9.1f}ms")
            except MemoryError:
                row.append("OOM")
        print(f"{n:>12} | {' | '.join(f'{r:>12}' for r in row)}")

    print("\n== experiment path: forks.sweep, 5 scenarios (interpreted, best of 3) ==")
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        t = bench_sweep(n)
        per_sc = t / 5
        print(f"{n:>12} rows: {t * 1000:9.1f}ms total, {per_sc * 1000:8.1f}ms/scenario")

    print("\n== param-only sweep via fused run() (no session; the fast path) ==")
    n = 1_000_000
    exe = Engine().bind(pipeline, mode="fused")
    df = pl.DataFrame({"income": [1000.0] * n, "debt": [float(i) for i in range(n)]})
    t = time.perf_counter()
    for i in range(5):
        exe.run(df, params={"demo": {"approved": {"limit": 0.1 + i * 0.1}}})
    print(f"{n:>12} rows, 5 scenarios: {(time.perf_counter() - t) * 1000:9.1f}ms total")

    print("\n== real pipeline: 02-affordability score() per record ==")
    try:
        t = bench_affordability_score()
        print(f"{t * 1e6:8.0f} us/record  ->  ~{1 / t:6.1f} records/s (interpreted)")
    except Exception as e:
        print(f"skipped: {e}")
