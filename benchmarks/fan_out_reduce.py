"""Fan-out/reduce (gap 4): score every bundle and keep the best, three ways.

Reproduces the running example from `notes/tmcfeval/DECIDER_GAPS.md` §4 now that
gap 3 (`Columnar[Item]`, `each`) has landed: an order carries line items, the
pipeline tries every bundle (subset) of items, prices a shipping offer from a
rate card (`param_table`), disqualifies bundles that are too heavy, scores the
rest, and keeps the best.

Three methods, one answer:

- hand-written `@njit` search — the ceiling, and the reference for correctness;
- a `decider.loop` written out by hand — the verbose thing users did before;
- `optimise(...)` — the library construct, which lowers to that same loop.

    uv run python benchmarks/fan_out_reduce.py [items] [orders]
"""
import gc
import statistics
import sys
import time
from typing import TypedDict

import numba
import numpy as np
import polars as pl

from decider import Columnar, Engine, Table, flow, loop, optimise, param, param_table, step

ITEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ORDERS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
MAX_WEIGHT = 30.0
LADDER = [{"floor": 0, "rate": 0.2}, {"floor": 10, "rate": 0.15}, {"floor": 20, "rate": 0.1}]


class Item(TypedDict):
    price: float
    weight: float


# --- the evaluate steps, shared by every method ---------------------------------

@step(output="bundle_weight")
def bundle_weight(index: int, items: Columnar[Item]) -> float:
    w = 0.0
    for j in range(len(items.weight)):
        if (index >> j) & 1:
            w += items.weight[j]
    return w


@step(output="bundle_total")
def bundle_total(index: int, items: Columnar[Item]) -> float:
    t = 0.0
    for j in range(len(items.price)):
        if (index >> j) & 1:
            t += items.price[j]
    return t


def rate(bundle_weight: float, ladder: Table = param_table({"floor": int, "rate": float},
                                                            default=LADDER)) -> float:
    r = 0.0
    for i in range(len(ladder.floor)):
        if bundle_weight >= ladder.floor[i]:
            r = ladder.rate[i]
    return r


@step(output="offer")
def offer(bundle_weight: float, rate: float) -> float:
    return bundle_weight * rate


@step(output="score")
def score(bundle_total: float, offer: float) -> float:
    return bundle_total - offer


def heavy(bundle_weight: float, max_weight: float = param(MAX_WEIGHT)) -> bool:
    return bundle_weight > max_weight


@step(output="count")
def n_bundles(items: Columnar[Item]) -> int:
    return (1 << len(items.price)) - 1


# --- method 0: hand-written numba (the ceiling) ---------------------------------

@numba.njit(cache=True, nogil=True)
def _hand(price, weight, ladder_floor, ladder_rate, max_weight):
    best_index = -1
    best_score = -1e18
    n = (1 << len(price)) - 1
    for index in range(1, n + 1):
        w = 0.0
        t = 0.0
        for j in range(len(price)):
            if (index >> j) & 1:
                w += weight[j]
                t += price[j]
        if w > max_weight:
            continue
        r = 0.0
        for i in range(len(ladder_floor)):
            if w >= ladder_floor[i]:
                r = ladder_rate[i]
        s = t - w * r
        if s > best_score:
            best_score, best_index = s, index
    return best_index, best_score


@numba.njit(cache=True, nogil=True)
def _hand_batch(price, weight, ladder_floor, ladder_rate, max_weight, out_index, out_score):
    for r in range(price.shape[0]):
        i, s = _hand(price[r], weight[r], ladder_floor, ladder_rate, max_weight)
        out_index[r], out_score[r] = i, s


# --- method 1: `decider.loop` by hand (what a user writes today) ----------------

def more(index: int, count: int) -> bool:
    return index <= count


@step(outputs=("best_index", "best_score", "evaluated", "disqualified"))
def keep(index: int, best_index: int, best_score: float, evaluated: int, disqualified: int,
         score: float, heavy: bool) -> tuple[int, float, int, int]:
    if heavy:
        disqualified += 1
    else:
        evaluated += 1
        if score > best_score:
            best_index, best_score = index, score
    return best_index, best_score, evaluated, disqualified


@step(output="index")
def advance(index: int) -> int:
    return index + 1


@step(outputs=("index", "best_index", "best_score", "evaluated", "disqualified"))
def seed(count: int) -> tuple[int, int, float, int, int]:
    return 1, -1, -1e18, 0, 0


def explicit_loop():
    body = flow(bundle_weight, bundle_total, rate, offer, score, heavy, keep, advance, name="body")
    search = loop(more, body,
                  carries=["index", "best_index", "best_score", "evaluated", "disqualified"],
                  max_iterations=(1 << ITEMS), name="search")
    return flow(n_bundles, seed, search, name="order")


# --- method 2: `optimise`, the library construct (lowers to the same loop) ------

OPT = optimise(n_bundles, flow(bundle_weight, bundle_total, rate, offer, score, name="evaluate"),
               disqualify=heavy, max_candidates=(1 << ITEMS), name="best")
PIPELINE = flow(OPT, name="order")

# --- data -----------------------------------------------------------------------

rng = np.random.default_rng(0)
prices = rng.uniform(1, 20, (ORDERS, ITEMS)).round(2)
weights = rng.uniform(1, 8, (ORDERS, ITEMS)).round(2)
FRAME = pl.DataFrame(
    {"items": [[{"price": float(p), "weight": float(w)} for p, w in zip(pr, wr)]
               for pr, wr in zip(prices, weights)]},
    schema={"items": pl.List(pl.Struct({"price": pl.Float64, "weight": pl.Float64}))})
RECORD = {"items": [{"price": float(p), "weight": float(w)} for p, w in zip(prices[0], weights[0])]}
LF = np.array([r["floor"] for r in LADDER], dtype=np.int64)
LR = np.array([r["rate"] for r in LADDER], dtype=np.float64)
pc = np.ascontiguousarray(prices)
wc = np.ascontiguousarray(weights)


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
          + (f"   {got}" if got is not None else ""))


if __name__ == "__main__":
    print(f"items={ITEMS} orders={ORDERS} masks={((1 << ITEMS) - 1)}")
    out_index = np.empty(ORDERS, np.int64)
    out_score = np.empty(ORDERS)
    _hand_batch(pc, wc, LF, LR, MAX_WEIGHT, out_index, out_score)
    ref = (int(out_index[0]), out_score[0])
    print(f"  reference (hand njit): best_index={ref[0]} score={ref[1]:.3f}")

    report("hand-written njit, 1 order", timed(lambda: _hand(pc[0], wc[0], LF, LR, MAX_WEIGHT), 100))
    report(f"hand-written njit, {ORDERS} orders",
           timed(lambda: _hand_batch(pc, wc, LF, LR, MAX_WEIGHT, out_index, out_score), 20))

    exe = Engine().bind(PIPELINE, mode="fused")
    got = exe.score(RECORD)
    assert got["best_index"] == ref[0] and abs(got["best_score"] - ref[1]) < 1e-6, (got, ref)
    print(f"  optimise (fused): best_index={got['best_index']} score={got['best_score']:.3f} "
          f"evaluated={got['evaluated']} disqualified={got['disqualified']}")
    print(f"  packed loops: {sorted(getattr(exe.runner, 'packed', {}))}")
    print(f"  fallbacks: {exe.runner.fallbacks()}")
    report("optimise score() 1 order", timed(lambda: exe.score(RECORD), 200))
    report(f"optimise run() {ORDERS} orders", timed(lambda: exe.run(FRAME), 10))

    loop_exe = Engine().bind(explicit_loop(), mode="fused")
    lg = loop_exe.score(RECORD)
    assert lg["best_index"] == got["best_index"] and abs(lg["best_score"] - got["best_score"]) < 1e-9
    report("explicit loop score() 1 order", timed(lambda: loop_exe.score(RECORD), 200))
    report(f"explicit loop run() {ORDERS} orders", timed(lambda: loop_exe.run(FRAME), 10))
