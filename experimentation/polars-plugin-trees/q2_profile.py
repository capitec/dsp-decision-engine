"""Where the polars-side time goes: each of the 4 expressions alone, 1M rows, in-process."""
import gc, time, polars as pl
from common import log
import decider_trees as dt, trees
df = trees.make_frame(1_000_000)
c = pl.col
full = trees.run_polars(df)  # has pts1, adj, pts2
cases = {
    "noop(1 col)": (df, dt.noop(c("income")).alias("n")),
    "plugin T1 walk_value (4 cols)": (df, dt.walk_value(*[c(f) for f in trees.T1_FEATURES], tree=trees.T1).alias("pts1")),
    "plugin T1 walk (Int32 leaf)": (df, dt.walk(*[c(f) for f in trees.T1_FEATURES], tree=trees.T1).alias("l")),
    "plugin T1 parallel=True": (df, dt.walk_value(*[c(f) for f in trees.T1_FEATURES], tree=trees.T1, parallel=True).alias("pts1")),
    "oracle T1 when/then": (df, dt.to_when_then(trees.T1, [c(f) for f in trees.T1_FEATURES], value=True).alias("pts1")),
    "arithmetic": (full, (c("pts1") * c("income") / 1000.0 + c("age")).alias("adj")),
    "plugin T2 walk_value (2 cols)": (full, dt.walk_value(*[c(f) for f in trees.T2_FEATURES], tree=trees.T2).alias("pts2")),
    "threshold": (full, (c("pts2") >= trees.CUTOFF).alias("approve")),
}
for name, (frame, e) in cases.items():
    frame.with_columns(e)
    ts = []
    for _ in range(5):
        gc.collect(); t0 = time.perf_counter(); frame.with_columns(e); ts.append(time.perf_counter() - t0)
    ts.sort(); med = ts[2]
    print(f"{name:34s} {med/1e6*1e9:8.1f} ns/row  (min {ts[0]*1e3:7.1f} ms, med {med*1e3:7.1f} ms)")
    log("q2_profile_1M", expr=name, ns_per_row_median=med / 1e6 * 1e9, wall_median_ms=med * 1e3, threads=pl.thread_pool_size())
