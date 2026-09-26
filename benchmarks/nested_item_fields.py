"""What an `Item` field costs: a `str` field, and iterating records against arrays per field.

`uv run python benchmarks/nested_item_fields.py`
"""
from __future__ import annotations

import statistics
import time
import warnings
from collections import namedtuple
from typing import TypedDict

import numpy as np
import polars as pl
from numba import njit

from decider import Engine, Rows, flow
from decider.engine.compile.rows import build_rows

N = 200_000
ITEMS = 3
CALLS = 3000
WORDS = ("snoop", "dogg", "privé", "𝄞clef")


class Num(TypedDict):
    el_1: int


class TwoNum(TypedDict):
    el_1: int
    el_3: float


class Str(TypedDict):
    el_1: int
    el_2: str


def find_num(items: Rows[Num]) -> int:
    for j in range(len(items.el_1)):
        if items.el_1[j] == 400:
            return j
    return -1


def find_two_num(items: Rows[TwoNum]) -> int:
    for j in range(len(items.el_1)):
        if items.el_1[j] == 400 and items.el_3[j] > 0.5:
            return j
    return -1


def find_str(items: Rows[Str]) -> int:
    for j in range(len(items.el_1)):
        if items.el_1[j] == 400 and items.el_2[j] == "snoop":
            return j
    return -1


def find_str_prefix(items: Rows[Str]) -> int:
    for j in range(len(items.el_1)):
        if items.el_1[j] == 400 and items.el_2[j].startswith("sno"):
            return j
    return -1


def find_list(items: list) -> int:
    for j in range(len(items)):
        if items[j]["el_1"] == 400 and items[j]["el_2"] == "snoop":
            return j
    return -1


def rows(n: int, items: int, fields: tuple[str, ...]) -> list[list[dict]]:
    rng = np.random.default_rng(0)
    el_1 = rng.choice([1, 400], n * items)
    el_2 = rng.choice(WORDS, n * items)
    el_3 = rng.random(n * items)
    out = []
    for i in range(n):
        row = []
        for j in range(items):
            k = i * items + j
            item = {}
            if "el_1" in fields:
                item["el_1"] = int(el_1[k])
            if "el_2" in fields:
                item["el_2"] = str(el_2[k])
            if "el_3" in fields:
                item["el_3"] = float(el_3[k])
            row.append(item)
        out.append(row)
    return out


def _time(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def batch(exe, df, reps=3) -> float:
    exe.run(df.head(8))
    return N / min(_time(lambda: exe.run(df)) for _ in range(reps))


def latency(cases: list[tuple]) -> list[tuple[float, float, float]]:
    # Interleaved: a busy box drifts every variant the same way, so p50s stay comparable.
    for exe, record in cases:
        for _ in range(200):
            exe.score(record)
    times: list[list[float]] = [[] for _ in cases]
    for _ in range(CALLS):
        for (exe, record), ts in zip(cases, times):
            t = time.perf_counter()
            exe.score(record)
            ts.append((time.perf_counter() - t) * 1e6)
    out = []
    for ts in times:
        ts.sort()
        out.append((statistics.median(ts), ts[int(0.99 * len(ts))], ts[0]))
    return out


def end_to_end() -> None:
    print("\n=== 1. a str Item field, end to end (200k rows, fused) ===")
    variants = (
        ("Rows[Item], one int field", find_num, ("el_1",), "fused"),
        ("Rows[Item], int + float", find_two_num, ("el_1", "el_3"), "fused"),
        ("Rows[Item], int + str, == literal", find_str, ("el_1", "el_2"), "fused"),
        ("Rows[Item], int + str, startswith", find_str_prefix, ("el_1", "el_2"), "fused"),
        ("list[dict], Python per row", find_list, ("el_1", "el_2"), "fused"),
        ("Rows[Item], int + str, interpreted", find_str, ("el_1", "el_2"), "interpreted"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases = []
        for _, fn, fields, mode in variants:
            data = rows(N, ITEMS, fields)
            df = pl.DataFrame({"items": data})
            exe = Engine().bind(flow(fn, name="p"), mode=mode)
            cases.append((exe, df, {"items": data[0]}))
        lats = latency([(exe, record) for exe, _, record in cases])
        out = [(label, batch(exe, df), *lat) for (label, *_), (exe, df, _), lat in zip(variants, cases, lats)]
    print(f"{'variant':44} {'rows/s':>10} {'p50 us':>8} {'p99 us':>8} {'min us':>8}")
    for label, rps, p50, p99, lo in out:
        print(f"{label:44} {rps / 1e6:9.2f}M {p50:8.1f} {p99:8.1f} {lo:8.1f}")


def build_costs() -> None:
    print("\n=== 2. build_rows alone (us per call) ===")
    cases = (("one int field", ("el_1",), (("el_1", int),)),
             ("int + float", ("el_1", "el_3"), (("el_1", int), ("el_3", float))),
             ("int + str", ("el_1", "el_2"), (("el_1", int), ("el_2", str))))
    print(f"{'schema':16} {'n':>8} {'python read':>12} {'arrow read':>12}")
    for n in (1, 256, 20_000, N):
        for label, fields, schema in cases:
            data = rows(n, ITEMS, fields)
            series = pl.Series("items", data)
            values = np.empty(n, object)
            for i, row in enumerate(data):
                values[i] = row
            py = min(_time(lambda: build_rows(values, schema, None, [])) for _ in range(3)) * 1e6
            ar = min(_time(lambda: build_rows(values, schema, series, [])) for _ in range(3)) * 1e6
            print(f"{label:16} {n:8} {py:12.1f} {ar if n >= 256 else float('nan'):12.1f}")


# ---- 3. arrays per field against a record array per row -----------------------------------------
# Both shapes take one argument per call, as a step does: `Rows[Item]`'s namedtuple of k arrays,
# or one interleaved record array. Bodies touch one field or all k, and either count items or
# find the first match, which is what the user's snippet does.
@njit(cache=True)
def _cols_one(items):
    t = 0.0
    for j in range(len(items.f0)):
        t += items.f0[j]
    return t


@njit(cache=True)
def _cols_all2(items):
    t = 0.0
    for j in range(len(items.f0)):
        t += items.f0[j] + items.f1[j]
    return t


@njit(cache=True)
def _cols_all8(items):
    t = 0.0
    for j in range(len(items.f0)):
        t += (items.f0[j] + items.f1[j] + items.f2[j] + items.f3[j]
              + items.f4[j] + items.f5[j] + items.f6[j] + items.f7[j])
    return t


@njit(cache=True)
def _cols_find(items):
    for j in range(len(items.f0)):
        if items.f0[j] > 0.999:
            return j
    return -1


@njit(cache=True)
def _recs_one(items):
    t = 0.0
    for i in items:
        t += i.f0
    return t


@njit(cache=True)
def _recs_all2(items):
    t = 0.0
    for i in items:
        t += i.f0 + i.f1
    return t


@njit(cache=True)
def _recs_all8(items):
    t = 0.0
    for i in items:
        t += i.f0 + i.f1 + i.f2 + i.f3 + i.f4 + i.f5 + i.f6 + i.f7
    return t


@njit(cache=True)
def _recs_find(items):
    for j in range(len(items)):
        if items[j].f0 > 0.999:
            return j
    return -1


def shapes() -> None:
    print("\n=== 3. Rows[Item] namedtuple against a record array per row (njit, no engine, us per row) ===")
    print(f"{'shape':34} {'1 field':>9} {'all k':>9} {'find':>9} {'build':>9}")
    for k in (2, 8):
        names = [f"f{i}" for i in range(k)]
        nt = namedtuple("Item", names)
        dt = np.dtype([(n, np.float64) for n in names], align=True)
        for items in (3, 30, 300, 3000):
            total = items * 1000
            rng = np.random.default_rng(0)
            flat = tuple(rng.random(total) for _ in range(k))
            cols_all = _cols_all2 if k == 2 else _cols_all8
            recs_all = _recs_all2 if k == 2 else _recs_all8

            def build_cols():
                return [nt(*(f[r * items:(r + 1) * items] for f in flat)) for r in range(1000)]

            def build_recs():
                out = np.empty(total, dt)
                for n, col in zip(names, flat):
                    out[n] = col
                return [out[r * items:(r + 1) * items] for r in range(1000)]

            col_rows, rec_rows = build_cols(), build_recs()

            def over(fn, rows):
                def go():
                    for row in rows:
                        fn(row)
                return go

            runs = {"cols_one": over(_cols_one, col_rows), "cols_all": over(cols_all, col_rows),
                    "cols_find": over(_cols_find, col_rows), "recs_one": over(_recs_one, rec_rows),
                    "recs_all": over(recs_all, rec_rows), "recs_find": over(_recs_find, rec_rows)}
            for go in runs.values():
                go()
            best = {name: min(_time(go) for _ in range(5)) * 1e6 / 1000 for name, go in runs.items()}
            bc = min(_time(build_cols) for _ in range(5)) * 1e6 / 1000
            br = min(_time(build_recs) for _ in range(5)) * 1e6 / 1000
            print(f"  k={k}, {items:4} items/row namedtuple  {best['cols_one']:9.2f} {best['cols_all']:9.2f} "
                  f"{best['cols_find']:9.2f} {bc:9.2f}")
            print(f"  {'':20} records     {best['recs_one']:9.2f} {best['recs_all']:9.2f} "
                  f"{best['recs_find']:9.2f} {br:9.2f}")


# ---- 4. the same two shapes with a str field, which is the user's own predicate -----------------
def with_strings() -> None:
    """A spike-only shape B: a record whose numpy dtype expands a span field into two int64 halves."""
    from numba import typeof
    from numba.core import types
    from numba.core.datamodel import models
    from numba.extending import register_model, typeof_impl

    from decider.engine.compile.span import SPAN, span_array

    class SpanRecord(types.Record):
        def __init__(self, fields, size, aligned, np_dtype):
            self._np_dtype = np_dtype
            super().__init__(fields, size, aligned)

        @property
        def dtype(self):
            return self._np_dtype

    register_model(SpanRecord)(models.RecordModel)
    mixed_dt = np.dtype([("el_1", "i8"), ("el_2_addr", "i8"), ("el_2_n", "i8")])
    rec_t = SpanRecord([("el_1", {"type": types.int64, "offset": 0, "alignment": 8, "title": None}),
                        ("el_2", {"type": SPAN, "offset": 8, "alignment": 8, "title": None})],
                       24, True, mixed_dt)

    class MixedArray(np.ndarray):
        pass

    @typeof_impl.register(MixedArray)
    def _typeof_mixed(val, c):
        return types.Array(rec_t, 1, "C")

    @njit(cache=False)
    def cols_find(items):
        for j in range(len(items.el_1)):
            if items.el_1[j] == 400 and items.el_2[j] == "snoop":
                return j
        return -1

    @njit(cache=False)
    def recs_find(items):
        for j in range(len(items)):
            i = items[j]
            if i.el_1 == 400 and i.el_2 == "snoop":
                return j
        return -1

    @njit(cache=False)
    def recs_iter(items):
        for i in items:
            if i.el_1 == 400 and i.el_2 == "snoop":
                return i.el_1
        return -1

    print("\n=== 4. the user's predicate with a str field (njit, us per row) ===")
    Item = namedtuple("Item", "el_1 el_2")
    for items in (3, 30, 300):
        total = items * 1000
        rng = np.random.default_rng(0)
        el_1 = rng.choice([1, 400], total).astype(np.int64)
        words = [str(w).encode() for w in rng.choice(WORDS, total)]
        pairs = np.empty(total * 2, np.int64)
        from decider.engine.compile.span import _address
        for j, b in enumerate(words):
            pairs[2 * j], pairs[2 * j + 1] = _address(b), len(b)
        col_spans = span_array(pairs, words)
        col_rows = [Item(el_1[r * items:(r + 1) * items], col_spans[r * items:(r + 1) * items])
                    for r in range(1000)]
        recs = np.empty(total, mixed_dt).view(MixedArray)
        recs["el_1"] = el_1
        recs["el_2_addr"] = pairs[0::2]
        recs["el_2_n"] = pairs[1::2]
        recs.kept = words
        rec_rows = [recs[r * items:(r + 1) * items].view(MixedArray) for r in range(1000)]
        for row in rec_rows:
            row.kept = words
        cols_find(col_rows[0])
        recs_find(rec_rows[0])
        recs_iter(rec_rows[0])
        assert typeof(rec_rows[0]) == types.Array(rec_t, 1, "C")

        def go(fn, rows):
            def run():
                for row in rows:
                    fn(row)
            return run

        best = {name: min(_time(go(fn, rows)) for _ in range(5)) * 1e6 / 1000
                for name, (fn, rows) in (("namedtuple items.el_2[j]", (cols_find, col_rows)),
                                         ("records items[j].el_2", (recs_find, rec_rows)),
                                         ("records for i in items", (recs_iter, rec_rows)))}
        print(f"  {items:4} items/row: " + "  ".join(f"{n} {v:6.2f}" for n, v in best.items()))


def main() -> None:
    end_to_end()
    build_costs()
    shapes()
    with_strings()


if __name__ == "__main__":
    main()
