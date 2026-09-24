"""Branch and loop batch throughput: fused (packed kernels) against stepped (Python-driven).

    uv run python benchmarks/control_flow.py
"""
import time

import numpy as np
import polars as pl

import decider
from decider.engine import Engine

N_BRANCH = 1_000_000
N_LOOP = 200_000


def is_high(income: float) -> bool:
    return income > 10000.0


def high_limit(limit: float, income: float) -> float:
    return min(limit, income * 3.0)


def low_limit(limit: float, income: float) -> float:
    return min(limit, income * 1.5)


def owing(balance: float) -> bool:
    return balance > 0.0


def pay(balance: float, payment: float) -> float:
    return balance * 1.01 - payment


rng = np.random.default_rng(0)
branch_frame = pl.DataFrame({"income": rng.uniform(3000, 20000, N_BRANCH), "limit": np.full(N_BRANCH, 40000.0)})
loop_frame = pl.DataFrame({"balance": rng.uniform(1000, 50000, N_LOOP), "payment": rng.uniform(600, 2000, N_LOOP)})


def ours(mode="fused"):
    limits = decider.branch(is_high, decider.step(high_limit, output="limit"), decider.step(low_limit, output="limit"),
                            modifies=["limit"], name="limits")
    repay = decider.loop(owing, decider.step(pay, output="balance"), carries=["balance"], max_iterations=360,
                         name="repay")
    return Engine().bind(decider.flow(limits), mode).run, Engine().bind(decider.flow(repay), mode).run


def rows_per_s(run, frame, reps=7):
    first = run(frame)
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        run(frame)
        best = min(best, time.perf_counter() - t)
    return frame.height / best, first


results = {}
outputs = {}
engines = {"decider fused": ours(), "decider stepped": ours("stepped")}
for name, (run_branch, run_loop) in engines.items():
    reps = 2 if "stepped" in name else 7
    b, out_b = rows_per_s(run_branch, branch_frame, reps)
    lp, out_l = rows_per_s(run_loop, loop_frame, reps)
    results[name] = (b, lp)
    outputs[name] = (out_b["limit"], out_l["balance"])

b2, l2 = outputs["decider fused"]
assert all(b.equals(b2) and lp.equals(l2) for b, lp in outputs.values()), "modes disagree"
print(f"{'engine':<18}{'branch rows/s':>16}{'loop rows/s':>16}")
for name, (b, lp) in results.items():
    print(f"{name:<18}{b:>16,.0f}{lp:>16,.0f}")
