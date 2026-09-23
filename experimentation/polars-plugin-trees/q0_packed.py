"""Packed wire format vs list-of-dicts: the per-call cost of shipping the tree, and end to end."""
import gc, sys, time, polars as pl
from common import log
import decider_trees as dt, trees
THREADS = pl.thread_pool_size(); c = pl.col
def bench(fn, reps=2000, warm=100):
    for _ in range(warm): fn()
    gc.disable(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); fn(); ts.append(time.perf_counter_ns() - t0)
    gc.enable(); ts.sort(); return {"p50_us": ts[len(ts)//2] / 1e3, "min_us": ts[0] / 1e3}
def row(name, r, **kw):
    print(f"  {name:62s} p50 {r['p50_us']:8.1f} us   min {r['min_us']:8.1f}")
    log("q0_packed", threads=THREADS, path=name, loadavg_1m=float(open("/proc/loadavg").read().split()[0]), **r, **kw)
f1 = trees.make_frame(1); cols4 = [c(f) for f in trees.T1_FEATURES]
big = trees.make_frame(20000)
assert big.with_columns(dt.walk_value_packed(*cols4, tree=trees.T1).alias("v"))["v"].equals(big.with_columns(dt.walk_value(*cols4, tree=trees.T1).alias("v"))["v"])
print(f"=== packed vs dicts, 1-row frame, threads={THREADS}")
e = dt.noop_kwargs(*cols4, tree=trees.T1).alias("n"); row("dicts : noop_kwargs T1 (17 nodes)", bench(lambda: f1.with_columns(e)))
e = dt.noop_packed(*cols4, tree=trees.T1).alias("n"); row("packed: noop_packed T1 (17 nodes)", bench(lambda: f1.with_columns(e)))
e = dt.walk_value(*cols4, tree=trees.T1).alias("n"); row("dicts : walk_value T1", bench(lambda: f1.with_columns(e)))
e = dt.walk_value_packed(*cols4, tree=trees.T1).alias("n"); row("packed: walk_value_packed T1", bench(lambda: f1.with_columns(e)))
def chain(depth):
    return [dt.test(0, "<", float(-i), depth, i + 1 if i + 1 < depth else depth + 1) for i in range(depth)] + [dt.leaf(0, 1.0), dt.leaf(1, 2.0)]
for depth in (64, 512):
    t = chain(depth)
    e1 = dt.walk_value(c("income"), tree=t).alias("n"); e2 = dt.walk_value_packed(c("income"), tree=t).alias("n")
    row(f"  chain {depth}: dicts  walk_value", bench(lambda: f1.with_columns(e1), reps=500), depth=depth)
    row(f"  chain {depth}: packed walk_value_packed", bench(lambda: f1.with_columns(e2), reps=500), depth=depth)
# end to end, single record, packed, one nested expression
record = f1.row(0, named=True); schema = f1.schema
pts1 = dt.walk_value_packed(*cols4, tree=trees.T1)
adj = pts1 * c("income") / 1000.0 + c("age")
nested = (dt.walk_value_packed(adj, c("sector"), tree=trees.T2) >= trees.CUTOFF).alias("approve")
row("E2E packed: DataFrame([record]) -> 1 nested expr -> item()", bench(lambda: pl.DataFrame([record], schema=schema).select(nested).item()))
row("packed: 1 nested expr on existing 1-row frame", bench(lambda: f1.select(nested)))
sys.path.insert(0, "/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/src")
p = trees.decider2_pipeline(); p.precompile()
row("decider2 score(record) (same process, same moment)", bench(lambda: p.score(record)))
