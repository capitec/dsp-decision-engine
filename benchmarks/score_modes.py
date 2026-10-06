"""Warm single-record score() p50 per mode, through the public API only.

Run: uv run python benchmarks/score_modes.py [label]
"""
import statistics, sys, time
from decider import Engine, flow, param

def ratio(income: float, debt: float) -> float:
    return debt / income

def affordable(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit

def offer(income: float, ratio: float, cap: float = param(5.0)) -> float:
    return income * cap * (1.0 - ratio)

pipeline = flow(ratio, affordable, offer, name="afford")
record = {"income": 1000.0, "debt": 200.0}
out = {}
for mode in ("interpreted", "stepped", "fused"):
    exe = Engine().bind(pipeline, mode=mode)
    for _ in range(2000):
        exe.score(record)
    best = []
    for _ in range(7):
        ts = []
        for _ in range(5000):
            t = time.perf_counter_ns(); exe.score(record); ts.append(time.perf_counter_ns() - t)
        best.append(statistics.median(ts) / 1000)
    out[mode] = min(best)
print(sys.argv[1] if len(sys.argv) > 1 else "", " ".join(f"{m}={v:.1f}us" for m, v in out.items()))
