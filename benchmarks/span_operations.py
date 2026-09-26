"""What the readable span form costs against a semantic str and the numeric ceiling.

    uv run python benchmarks/span_operations.py
"""
import gc
import time
import warnings

import numpy as np
import polars as pl

from decider import Raw, flow, param, python_only
from decider.engine import Engine

N = 200_000
CALLS = 3000
SECTORS = ["private", "public", "government", "informal"]
rng = np.random.default_rng(0)
FRAME = pl.DataFrame({"sector": rng.choice(SECTORS, N), "amount": rng.uniform(0, 1e5, N)})
SECTOR = {"sector": "private"}
AMOUNT = {"amount": 4100.0}


def semantic_str(sector: str) -> bool:
    return sector == "private"


@python_only
def semantic_str_python(sector: str) -> bool:
    return sector == "private"


def span_readable(sector: Raw[bytes]) -> bool:
    return sector == "private"


def span_readable_param(sector: Raw[bytes], want: str = param("private")) -> bool:
    return sector == want


def span_index(sector: Raw[bytes]) -> bool:
    return sector[1] >= 0


def numeric(amount: float) -> bool:
    return amount >= 50000.0


def _time(f, *args):
    t = time.perf_counter()
    f(*args)
    return time.perf_counter() - t


def rows_per_s(run, frame, reps=7):
    run(frame)
    return N / min(_time(run, frame) for _ in range(reps))


def latency(score, row):
    for _ in range(500):
        score(row)
    gc.collect()
    samples = sorted(_time(score, row) for _ in range(CALLS))
    return samples[len(samples) // 2] * 1e6, samples[int(len(samples) * 0.99)] * 1e6


cases = [("numeric kernel (the ceiling)", numeric, AMOUNT),
         ("str, njit per row", semantic_str, SECTOR),
         ("str, Python per row", semantic_str_python, SECTOR),
         ("span, == literal", span_readable, SECTOR),
         ("span, == str param", span_readable_param, SECTOR),
         ("span, sector[1] >= 0", span_index, SECTOR)]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    engines = [(name, Engine().bind(flow(fn, name="p"), mode="fused"), row) for name, fn, row in cases]
    # Latency first: a batch just before leaves the allocator churning.
    results = [(name, *latency(exe.score, row), exe, row) for name, exe, row in engines]
    results = [(name, rows_per_s(exe.run, FRAME.select(list(row))), p50, p99) for name, p50, p99, exe, row in results]

print(f"{'variant':<30}{'batch rows/s':>16}{'score p50 us':>14}{'score p99 us':>14}")
for name, batch, p50, p99 in results:
    print(f"{name:<30}{batch:>16,.0f}{p50:>14.1f}{p99:>14.1f}")
