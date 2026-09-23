"""Where fused `run()` spends its time, and score()/one-row run latency, for three pipelines.

    uv run python benchmarks/zero_copy_spike.py [flagship|wide|tree ...]

Phases of one `run(df)`: `extract` (State.from_frame: Arrow export and column
reads), `driver` (reading state, null policies, bundles), `kernel` (the numba
call), `output` (Series per output + hstack). `DECIDER_ZERO_COPY=1` switches
the kernels to reading nullable/bool Arrow columns in place.
"""
import gc
import sys
import time

import numpy as np
import polars as pl

from decider import flow, missing_as, param, step
from decider.engine import Engine
from decider.engine.compile import units as U
from decider.engine.run import engine as E
from decider.engine.run import state as S

rng = np.random.default_rng(0)


# ---- flagship ----
def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float = param(48.0, ge=6, le=60),
                       income_threshold: float = param(5000.0, ge=0)) -> float:
    if min_net_salary < income_threshold:
        return min(term_cap, cap)
    return term_cap


def flagship(n):
    df = pl.DataFrame({
        "net_income": rng.uniform(3000, 20000, n), "expenses": rng.uniform(500, 3000, n),
        "instalment": rng.uniform(200, 3000, n), "term_cap": np.full(n, 60.0),
        "min_net_salary": rng.uniform(3000, 20000, n),
    })
    return flow(disposable_income, affordability_ratio, cap_by_income_band), df


# ---- wide: 80 float inputs, the second half nullable (10% nulls), 20 steps of 4 inputs, plus bools ----
def combo(a: float, b: float, flag: bool, d: float | None, c: float = missing_as(0.0)) -> float:
    base = a * 0.5 + b * 0.25 + c
    if d is not None:
        base += d
    return base + 1.0 if flag else base


def wide(n, steps=20):
    cols = {}
    for k in range(steps):
        cols[f"a{k}"] = rng.uniform(0, 1, n)
        cols[f"b{k}"] = rng.uniform(0, 1, n)
    df = pl.DataFrame(cols)
    nullable = {}
    for k in range(steps):
        for name in (f"c{k}", f"d{k}"):
            x = rng.uniform(0, 1, n)
            nullable[name] = pl.Series(name, x).scatter(np.flatnonzero(rng.uniform(0, 1, n) < 0.1), None)
    flags = {f"f{k}": rng.uniform(0, 1, n) < 0.5 for k in range(4)}
    df = df.with_columns(list(nullable.values())).with_columns(pl.Series(k, v) for k, v in flags.items())
    parts = []
    for k in range(steps):
        s = step(combo, name=f"combo{k}", output=f"out{k}").relabel(
            reads={"a": f"a{k}", "b": f"b{k}", "c": f"c{k}", "d": f"d{k}", "flag": f"f{k % 4}"})
        parts.append(s)
    return flow(*parts, name="wide"), df


FLOATS = [f"f{i}" for i in range(16)]
INTS = ["months", "enquiries"]


def _document(depth: int = 6) -> dict:
    # The tree of benchmarks/trees_vs_decider2.py, without importing decider2.
    trng = np.random.default_rng(0)
    nodes, edges, leaves = [], [], []

    def feature():
        return str(trng.choice(FLOATS + INTS))

    def threshold(f):
        return int(trng.integers(0, 24)) if f in INTS else round(float(trng.uniform(0.2, 0.8)), 3)

    def grow(d: int) -> str:
        nid = f"n{len(nodes)}"
        if d == depth:
            nodes.append({"id": nid, "data": {"type": "leaf", "result_idx": len(leaves)}})
            leaves.append(len(leaves))
            return nid
        shape = trng.integers(3)
        if shape == 0:
            f = feature()
            data = {"type": "unary", "condition": {"op": ">=", "feature": f, "threshold": threshold(f)}}
        elif shape == 1:
            conds = []
            for _ in range(int(trng.integers(2, 4))):
                f = feature()
                conds.append({"op": "<", "feature": f, "threshold": threshold(f)})
            data = {"type": "composite", "op": "and", "conditions": conds}
        else:
            f = str(trng.choice(FLOATS))
            data = {"type": "unary", "condition": {"op": "between", "feature": f, "min": 0.1, "max": 0.7}}
        nodes.append({"id": nid, "data": data})
        for k in range(2):
            edges.append({"source": nid, "target": grow(d + 1), "data": {"sourceIndex": [k]}})
        return nid

    grow(0)
    rows = [{"label": f"segment_{i % 7}", "score": float(i) * 1.5, "band": i % 5} for i in leaves]
    return {"name": "campaign", "nodes": nodes, "edges": edges,
            "output": {"data": rows, "default": {"label": "none", "score": 0.0, "band": -1},
                       "dtypes": [["label", "String"], ["score", "Float64"], ["band", "Int64"]]}}


def tree(n):
    from decider.steps.trees import TreeConfig

    df = pl.DataFrame({**{f: rng.uniform(0, 1, n) for f in FLOATS}, **{f: rng.integers(0, 24, n) for f in INTS}})
    return TreeConfig(name="campaign", tree=_document(), feature_types={f: "int" for f in INTS}), df


# ---- phase timers ----
T = {"extract": 0.0, "driver": 0.0, "kernel": 0.0, "output": 0.0}


def _time(f, *a):
    t = time.perf_counter()
    f(*a)
    return time.perf_counter() - t


def breakdown(exe, df, reps):
    orig_from = S.State.from_frame
    orig_run = U.Kernel.run
    orig_out = E.Executable.output
    S.State.from_frame = classmethod(lambda cls, *a, **k: _timed("extract", orig_from.__func__, cls, *a, **k))

    def krun(self, *a, **k):
        fn = self.fn
        self.fn = lambda *x: _timed("kernel", fn, *x)
        try:
            return orig_run(self, *a, **k)
        finally:
            self.fn = fn

    U.Kernel.run = krun
    E.Executable.output = lambda self, st: _timed("output", orig_out, self, st)
    try:
        exe.run(df)
        best = None
        for _ in range(reps):
            for k in T:
                T[k] = 0.0
            total = _time(exe.run, df)
            if best is None or total < best[0]:
                best = (total, dict(T))
    finally:
        S.State.from_frame, U.Kernel.run, E.Executable.output = orig_from, orig_run, orig_out
    total, t = best
    t["driver"] = total - sum(t.values())
    return total, t


def _timed(key, f, *a, **k):
    t = time.perf_counter()
    try:
        return f(*a, **k)
    finally:
        T[key] += time.perf_counter() - t


def latency(f, arg, calls):
    for _ in range(500):
        f(arg)
    gc.collect()
    s = sorted(_time(f, arg) for _ in range(calls))
    return s[len(s) // 2] * 1e6, s[int(len(s) * 0.99)] * 1e6


BUILDERS = {"flagship": flagship, "wide": wide, "tree": tree}


def main(names):
    for name in names:
        pipeline, big = BUILDERS[name](1_000_000)
        exe = Engine().bind(pipeline, mode="fused")
        one = big.head(1)
        record = one.row(0, named=True)
        exe.run(big.head(5000))
        p50, p99 = latency(exe.score, record, 20000)
        r50, r99 = latency(exe.run, one, 5000)
        print(f"{name}: score p50 {p50:.1f} p99 {p99:.1f} us | run(1 row) p50 {r50:.1f} p99 {r99:.1f} us")
        for n, reps in ((100_000, 15), (1_000_000, 7)):
            df = big.head(n)
            total, t = breakdown(exe, df, reps)
            parts = " ".join(f"{k} {v * 1e3:.2f}" for k, v in t.items())
            print(f"  {n:>9,} rows: {total * 1e3:.2f} ms  ({parts} ms)")


if __name__ == "__main__":
    main(sys.argv[1:] or list(BUILDERS))
