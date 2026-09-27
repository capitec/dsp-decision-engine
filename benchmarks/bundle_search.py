"""A `decider.loop` over bundle masks whose body sums the bundle's items, three ways.

The body reads `items: Columnar[Item]`, sliced out of flat per-field arrays inside the
loop's own kernel, so the loop packs. The hand-written `@njit` search is the ceiling.

    uv run python benchmarks/bundle_search.py [items] [orders]
"""
import gc
import statistics
import sys
import time
import warnings
from typing import TypedDict

import numba
import numpy as np
import polars as pl

from decider import Engine, Columnar, flow, loop, step

ITEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 9
ORDERS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
MASKS = (1 << ITEMS) - 1


class Item(TypedDict):
    price: float
    weight: float


def more(mask: int) -> bool:
    return mask <= MASKS


@step(output="total")
def bundle_total(mask: int, items: Columnar[Item]) -> float:
    total = 0.0
    for j in range(len(items.price)):
        if (mask >> j) & 1:
            total += items.price[j] * items.weight[j]
    return total


@step(output="best")
def keep_best(best: float, total: float) -> float:
    return total if total > best else best


@step(output="mask")
def advance(mask: int) -> int:
    return mask + 1


search = loop(more, flow(bundle_total, keep_best, advance, name="body"),
              carries=["mask", "best"], max_iterations=MASKS + 1, name="search")
pipeline = flow(search, name="order")

rng = np.random.default_rng(0)
prices = rng.uniform(1, 100, (ORDERS, ITEMS)).round(2)
weights = rng.uniform(0.1, 5, (ORDERS, ITEMS)).round(2)
FRAME = pl.DataFrame(
    {"items": [[{"price": float(p), "weight": float(w)} for p, w in zip(pr, wr)]
               for pr, wr in zip(prices, weights)],
     "mask": [1] * ORDERS, "best": [0.0] * ORDERS},
    schema={"items": pl.List(pl.Struct({"price": pl.Float64, "weight": pl.Float64})),
            "mask": pl.Int64, "best": pl.Float64})
RECORD = {"items": [{"price": float(p), "weight": float(w)} for p, w in zip(prices[0], weights[0])],
          "mask": 1, "best": 0.0}


@numba.njit(cache=True, nogil=True)
def hand_written(price, weight, masks):
    best = 0.0
    for mask in range(1, masks + 1):
        total = 0.0
        for j in range(len(price)):
            if (mask >> j) & 1:
                total += price[j] * weight[j]
        if total > best:
            best = total
    return best


@numba.njit(cache=True, nogil=True)
def hand_written_batch(price, weight, masks, out):
    for r in range(price.shape[0]):
        out[r] = hand_written(price[r], weight[r], masks)


def timed(fn, repeat, warm=2):
    for _ in range(warm):
        fn()
    gc.disable()
    times = []
    for _ in range(repeat):
        t = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t) / 1e6)
    gc.enable()
    return times


def p(times, q):
    return statistics.quantiles(times, n=100)[q - 1] if len(times) > 20 else max(times)


def report(label, times, got=None):
    print(f"  {label:<34} p50 {statistics.median(times):9.3f} ms   p99 {p(times, 99):9.3f} ms"
          + (f"   best={got:.2f}" if got is not None else ""))


def bench(mode, repeat_score, repeat_run):
    warnings.simplefilter("ignore")
    exe = Engine().bind(pipeline, mode=mode)
    runner = exe.runner
    exe.score(RECORD)
    print(f"[{mode}] packed loops: {sorted(getattr(runner, 'packed', {}))}")
    print(f"[{mode}] fallbacks: {runner.fallbacks()}")
    want = hand_written(np.ascontiguousarray(prices[0]), np.ascontiguousarray(weights[0]), MASKS)
    got = exe.score(RECORD)["best"]
    assert abs(got - want) < 1e-9, (got, want)
    report(f"{mode} score() 1 order", timed(lambda: exe.score(RECORD), repeat_score), got=got)
    out = exe.run(FRAME)
    assert abs(out["best"][0] - want) < 1e-9
    report(f"{mode} run() {ORDERS} orders", timed(lambda: exe.run(FRAME), repeat_run))


print(f"items={ITEMS} masks={MASKS} orders={ORDERS}")
pc = np.ascontiguousarray(prices)
wc = np.ascontiguousarray(weights)
out = np.empty(ORDERS)
hand_written_batch(pc, wc, MASKS, out)
report("hand-written njit, 1 order", timed(lambda: hand_written(pc[0], wc[0], MASKS), 30),
       got=hand_written(pc[0], wc[0], MASKS))
report(f"hand-written njit, {ORDERS} orders", timed(lambda: hand_written_batch(pc, wc, MASKS, out), 10))
for mode in ("stepped", "fused"):
    bench(mode, 20 if MASKS < 5000 else 5, 5 if MASKS < 5000 else 2)
