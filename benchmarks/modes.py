"""Flagship batch throughput and score() latency in every mode.

    uv run python benchmarks/modes.py
"""
import gc
import time

import numpy as np
import polars as pl

from decider import flow, param
from decider.engine import Engine

N = 1_000_000
ROW = {"net_income": 4100.0, "expenses": 1500.0, "instalment": 800.0, "term_cap": 60.0, "min_net_salary": 4100.0}


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float = param(48.0, ge=6, le=60),
                       income_threshold: float = param(5000.0, ge=0)) -> float:
    if min_net_salary < income_threshold:
        return min(term_cap, cap)
    return term_cap


flagship = flow(disposable_income, affordability_ratio, cap_by_income_band)
rng = np.random.default_rng(0)
frame = pl.DataFrame({
    "net_income": rng.uniform(3000, 20000, N), "expenses": rng.uniform(500, 3000, N),
    "instalment": rng.uniform(200, 3000, N), "term_cap": np.full(N, 60.0),
    "min_net_salary": rng.uniform(3000, 20000, N),
})


def rows_per_s(run, reps=7):
    run(frame)
    best = min(_time(run, frame) for _ in range(reps))
    return N / best


def _time(f, *args):
    t = time.perf_counter()
    f(*args)
    return time.perf_counter() - t


def latency(score, calls=20000):
    for _ in range(1000):
        score(ROW)
    gc.collect()
    samples = sorted(_time(score, ROW) for _ in range(calls))
    return samples[len(samples) // 2] * 1e6, samples[int(len(samples) * 0.99)] * 1e6


# Latency first: a million-row batch just before leaves the allocator churning.
engines = {}
for mode in ("fused", "stepped", "interpreted"):
    exe = Engine().bind(flagship, mode=mode)
    engines[f"decider {mode}"] = (exe.run, exe.score)
latencies = {name: latency(score, calls=2000 if "interpreted" in name else 20000)
             for name, (_, score) in engines.items()}
results = {name: (rows_per_s(run, reps=1 if "interpreted" in name else 7), *latencies[name])
           for name, (run, _) in engines.items()}

print(f"{'engine':<22}{'batch rows/s':>16}{'score p50 us':>14}{'score p99 us':>14}")
for name, (batch, p50, p99) in results.items():
    print(f"{name:<22}{batch:>16,.0f}{p50:>14.1f}{p99:>14.1f}")
