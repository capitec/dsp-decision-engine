"""THE PRIMARY MEASUREMENT: single-record and small batches, microseconds per CALL.
Run in a fresh process per thread count: POLARS_MAX_THREADS=1 python q0_single.py ; python q0_single.py
"""
from __future__ import annotations
import gc, pickle, sys, time
import polars as pl
from common import log
import decider_trees as dt
sys.path.insert(0, "/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/src")
import trees

THREADS = pl.thread_pool_size()
load = lambda: float(open("/proc/loadavg").read().split()[0])

def bench(fn, reps=3000, warm=100):
    for _ in range(warm): fn()
    gc.disable(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); fn(); ts.append(time.perf_counter_ns() - t0)
    gc.enable(); ts.sort()
    return {"p50_us": ts[len(ts)//2] / 1e3, "min_us": ts[0] / 1e3, "p99_us": ts[int(len(ts)*0.99)] / 1e3}

def row(name, r, n=1, group=""):
    print(f"  {name:58s} p50 {r['p50_us']:9.1f} us   min {r['min_us']:8.1f}   p99 {r['p99_us']:9.1f}")
    log("q0_single", threads=THREADS, batch=n, group=group, path=name, loadavg_1m=load(), **r)

c = pl.col
p = trees.decider2_pipeline(); p.precompile()
frames = {n: trees.make_frame(n) for n in (1, 10, 100, 1000)}
record = frames[1].row(0, named=True)
schema = frames[1].schema

# ---- the plugin's four passes, and the same logic composed as ONE nested expression
four = trees.polars_exprs()
T1 = trees.T1; T2 = trees.T2
pts1 = dt.walk_value(*[c(f) for f in trees.T1_FEATURES], tree=T1)
adj = pts1 * c("income") / 1000.0 + c("age")
pts2 = dt.walk_value(adj, c("sector"), tree=T2)
one_nested = (pts2 >= trees.CUTOFF).alias("approve")
def four_passes(df):
    for e in four: df = df.with_columns(e)
    return df

print(f"\n=== batch 1, POLARS threads={THREADS}, load {load():.1f}")
print("-- the two systems, end to end (dict in, answer out)")
row("decider2 score(record)                      [dict -> dict]", bench(lambda: p.score(record)), group="e2e")
row("plugin: DataFrame([record]) -> 4 passes -> row(0)", bench(lambda: four_passes(pl.DataFrame([record], schema=schema)).row(0, named=True)), group="e2e")
row("plugin: DataFrame([record]) -> 1 nested expr -> item()", bench(lambda: pl.DataFrame([record], schema=schema).select(one_nested).item()), group="e2e")
print("-- the plugin path, in parts")
row("Expr construction: walk_value(...) for T1 (pickles 17 nodes)", bench(lambda: dt.walk_value(*[c(f) for f in trees.T1_FEATURES], tree=T1)), group="parts")
row("  of which pickle.dumps(kwargs)", bench(lambda: pickle.dumps({"nodes": T1, "parallel": False}, protocol=5)), group="parts")
row("pl.DataFrame([record], schema)  (dict -> 1-row frame)", bench(lambda: pl.DataFrame([record], schema=schema)), group="parts")
row("frame.row(0, named=True)        (1-row frame -> dict)", bench(lambda: frames[1].row(0, named=True)), group="parts")
f1 = frames[1]
row("floor: df.with_columns(pl.lit(1))  planner+dispatch, no kernel", bench(lambda: f1.with_columns(pl.lit(1).alias("k"))), group="parts")
row("floor: df.select(pl.col('income')) ", bench(lambda: f1.select(c("income"))), group="parts")
row("floor: df.with_columns(income*2.0) native kernel", bench(lambda: f1.with_columns((c("income") * 2.0).alias("y"))), group="parts")
noop = dt.noop(c("income"))
row("plugin noop (prebuilt Expr): dlopen'd fn, zero work", bench(lambda: f1.with_columns(noop.alias("n"))), group="parts")
one_tree = dt.walk_value(*[c(f) for f in trees.T1_FEATURES], tree=T1).alias("pts1")
row("plugin T1 walk_value (prebuilt Expr), 1 row", bench(lambda: f1.with_columns(one_tree)), group="parts")
row("plugin T1 walk_value, Expr rebuilt every call", bench(lambda: f1.with_columns(dt.walk_value(*[c(f) for f in trees.T1_FEATURES], tree=T1).alias("pts1"))), group="parts")
row("4 passes on an existing 1-row frame", bench(lambda: four_passes(f1)), group="parts")
row("1 nested expr on an existing 1-row frame (select)", bench(lambda: f1.select(one_nested)), group="parts")
row("1 nested expr, lazy().select().collect()", bench(lambda: f1.lazy().select(one_nested).collect()), group="parts")
print("-- decider2 in parts, for the same comparison")
row("decider2 apply(1-row frame)", bench(lambda: p.apply(f1, mode="fused")), group="parts")

for n in (10, 100, 1000):
    f = frames[n]
    print(f"\n=== batch {n}, threads={THREADS}, load {load():.1f}")
    row(f"decider2 apply({n} rows)", bench(lambda: p.apply(f, mode="fused"), reps=1000), n=n, group="batch")
    row(f"plugin 4 passes ({n} rows)", bench(lambda: four_passes(f), reps=1000), n=n, group="batch")
    row(f"plugin 1 nested expr ({n} rows)", bench(lambda: f.select(one_nested), reps=1000), n=n, group="batch")
    row(f"plugin 1 nested expr, streaming ({n} rows)", bench(lambda: f.lazy().select(one_nested).collect(engine="streaming"), reps=300), n=n, group="batch")
    row(f"when/then oracle 4 passes ({n} rows)", bench(lambda: trees.run_polars(f, oracle=True), reps=300), n=n, group="batch")
# correctness of the nested form
big = trees.make_frame(20000)
assert big.select(one_nested)["approve"].equals(four_passes(big)["approve"])
print("\nnested expr == four passes on 20k rows: ok")
