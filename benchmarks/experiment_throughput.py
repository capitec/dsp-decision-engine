"""Experiment runner throughput: a baseline plus one param scenario, global vs per-node metrics.

uv run python benchmarks/experiment_throughput.py
"""
import time

import numpy as np
import polars as pl

from decider import flow, param
from decider.experiments import Divergence, Experiment, ValueCounts


def ratio(income: float, debt: float) -> float:
    return debt / income if income > 0 else 1.0


def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


def tier(approved: bool, income: float) -> float:
    return 1.0 if approved else (0.5 if income > 0 else 0.0)


PIPELINE = flow(ratio, approved, tier, name="demo")
SETUPS = {
    "global metric (fused)": lambda e: e.metric(Divergence(column="tier")),
    "node metrics (stepped)": lambda e: e.metric(Divergence(column="tier"),
                                                 ValueCounts(column="approved", nodes="demo/approved"),
                                                 Divergence(name="ratio", column="ratio", nodes="demo/ratio")),
}


def main():
    print(f"{'records':>12} " + " ".join(f"{name:>26}" for name in SETUPS))
    for n in (1_000, 100_000, 1_000_000, 10_000_000):
        rng = np.random.default_rng(0)
        df = pl.DataFrame({"income": rng.uniform(100, 5000, n), "debt": rng.uniform(0, 3000, n)})
        cells = []
        for setup in SETUPS.values():
            exp = setup(Experiment("bench", PIPELINE, df).scenario(
                "limit_0.3", params={"demo": {"approved": {"limit": 0.3}}}))
            start = time.perf_counter()
            exp.run()
            dt = time.perf_counter() - start
            cells.append(f"{dt * 1000:>9,.0f} ms {n / dt:>11,.0f}/s")
        print(f"{n:>12,} " + " ".join(f"{c:>26}" for c in cells))


if __name__ == "__main__":
    main()
