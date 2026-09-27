"""`Columnar[Item]` against plain `list[dict]` and a numeric-kernel ceiling.

`Columnar[Item]` is the opt-in fast path for a `list[dict]` column, so the bar it
has to clear is the plain Python step, on `score()` first. Also compares three
ways to build the per-row namedtuples in one process: the value-by-value loop
this replaced, the Python-values path, and the Arrow read.

    uv run python benchmarks/nested_rows.py [rows] [items]
"""
import gc
import sys
import time
from collections import namedtuple
from typing import TypedDict

import numpy as np
import polars as pl

from decider import Engine, Columnar, flow
from decider.engine.compile.rows import build_rows
from decider.engine.run.state import from_series

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
ITEMS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SCHEMA = (("price", float), ("qty", int))


class Item(TypedDict):
    price: float
    qty: int


def rows_total(items: Columnar[Item]) -> float:
    total = 0.0
    for j in range(len(items.price)):
        total += items.price[j] * items.qty[j]
    return total


def list_total(items: list) -> float:
    total = 0.0
    for item in items:
        total += item["price"] * item["qty"]
    return total


def numeric_total(a: float, b: float, c: float) -> float:
    return a + b + c


rng = np.random.default_rng(0)
prices = rng.uniform(1, 100, (N, ITEMS)).round(2)
qtys = rng.integers(1, 5, (N, ITEMS))
FRAME = pl.DataFrame(
    {"items": [[{"price": float(p), "qty": int(q)} for p, q in zip(pr, qr)]
               for pr, qr in zip(prices, qtys)]},
    schema={"items": pl.List(pl.Struct({"price": pl.Float64, "qty": pl.Int64}))})
NUMERIC = pl.DataFrame({"a": prices[:, 0], "b": prices[:, 1], "c": prices[:, 2]})
RECORD = {"items": [{"price": float(p), "qty": int(q)} for p, q in zip(prices[0], qtys[0])]}
NUMERIC_RECORD = {"a": 2.0, "b": 3.0, "c": 5.0}


V0_NT = namedtuple("V0_price_qty", ["price", "qty"])


def build_rows_v0(values, schema):
    """What `build_rows` was: one numpy store per field per item, then a Python slice loop."""
    dtypes = {float: np.float64, int: np.int64, bool: np.bool_}
    lengths = np.fromiter((0 if row is None else len(row) for row in values), np.int64, len(values))
    offsets = np.zeros(len(values) + 1, np.int64)
    np.cumsum(lengths, out=offsets[1:])
    fields = {name: np.empty(int(offsets[-1]), dtypes[t]) for name, t in schema}
    k = 0
    for row in values:
        if row is None:
            continue
        for item in row:
            for name, _ in schema:
                fields[name][k] = item[name]
            k += 1
    out = np.empty(len(values), object)
    for i in range(len(values)):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        out[i] = V0_NT(*(fields[name][lo:hi] for name, _ in schema))
    return out


def _time(fn, *a):
    t = time.perf_counter()
    fn(*a)
    return time.perf_counter() - t


def best(fn, *a, budget=0.05):
    """Seconds per call: the fastest of several rounds, each long enough to time cleanly."""
    fn(*a)
    gc.collect()
    gc.disable()
    try:
        reps = max(1, int(budget / max(_time(fn, *a), 1e-9)))
        return min(_time(_repeat, fn, a, reps) for _ in range(3)) / reps
    finally:
        gc.enable()


def _repeat(fn, a, reps):
    for _ in range(reps):
        fn(*a)


def latency(exe, record, calls=3000):
    for _ in range(300):
        exe.score(record)
    gc.collect()
    us = np.empty(calls)
    for k in range(calls):
        t = time.perf_counter()
        exe.score(record)
        us[k] = (time.perf_counter() - t) * 1e6
    return np.percentile(us, 50), np.percentile(us, 99)


def end_to_end():
    print(f"{N} rows x {ITEMS} items")
    print(f"{'variant':<34}{'batch rows/s':>14}{'score p50':>12}{'p99':>8}")
    for name, step, frame, record in (
        ("numeric kernel (the ceiling)", numeric_total, NUMERIC, NUMERIC_RECORD),
        ("list[dict], Python per row", list_total, FRAME, RECORD),
        ("Columnar[Item], compiled per row", rows_total, FRAME, RECORD),
    ):
        exe = Engine().bind(flow(step, name="p"), mode="stepped")
        p50, p99 = latency(exe, record)
        print(f"{name:<34}{frame.height / best(exe.run, frame, budget=2.0):>14,.0f}{p50:>11.1f}us{p99:>8.1f}")


def builders():
    """The three builders at every size; the Arrow read is forced, to show where it wins."""
    from decider.engine.compile import rows as rows_mod

    threshold = rows_mod.ARROW_ROWS
    print(f"\n{'rows':>8}{'v0 (copy loop)':>18}{'python values':>16}{'arrow':>10}{'from_series':>14}")
    for n in (1, 8, 64, 256, 4096, N):
        frame = FRAME.head(n)
        series = frame.get_column("items")
        values, _ = from_series(series)
        rows_mod.ARROW_ROWS = threshold
        us = [best(build_rows_v0, values, SCHEMA) * 1e6,
              best(build_rows, values, SCHEMA) * 1e6]
        rows_mod.ARROW_ROWS = 1
        try:
            alive: list = []
            assert build_rows(values, SCHEMA, series, alive) is not None and alive
            us.append(best(lambda: build_rows(values, SCHEMA, series, [])) * 1e6)
        finally:
            rows_mod.ARROW_ROWS = threshold
        us.append(best(from_series, series) * 1e6)
        print(f"{n:>8}" + "".join(f"{x:>{w},.0f}us" for x, w in zip(us, (16, 14, 8, 12))))


if __name__ == "__main__":
    end_to_end()
    builders()
