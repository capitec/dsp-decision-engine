"""Q4: what breaks or gets harder.
  (a) single record: decider2 score() vs the 4 plugin expressions on a 1-row frame
  (b) equivalence: plugin == pure-polars when/then oracle == decider2 (all rungs), on random frames
  (c) retune without recompiling: per-call cost of getting the tree across (pickle + serde-pickle), by tree size
  (d) errors, panics: run in subprocesses
"""
from __future__ import annotations
import gc, pickle, signal, subprocess, sys, time, statistics as st
import polars as pl
from common import log, HERE
import decider_trees as dt
sys.path.insert(0, "/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/src")
import trees

PY = sys.executable

def bench(fn, reps=2000, warm=50):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); fn(); ts.append(time.perf_counter_ns() - t0)
    ts.sort()
    return {"min_us": ts[0] / 1e3, "p50_us": ts[len(ts)//2] / 1e3, "p99_us": ts[int(len(ts)*0.99)] / 1e3}

# ---------------------------------------------------------------- (a) single record
frame1 = trees.make_frame(1)
record = frame1.row(0, named=True)
p = trees.decider2_pipeline()
p.precompile()
r_score = bench(lambda: p.score(record))
r_apply1 = bench(lambda: p.apply(frame1, mode="fused"))
exprs = trees.polars_exprs()
def plugin_1row_eager():
    df = frame1
    for e in exprs: df = df.with_columns(e)
    return df
r_plugin_eager = bench(plugin_1row_eager)
r_plugin_lazy = bench(lambda: frame1.lazy().with_columns(exprs[0]).with_columns(exprs[1]).with_columns(exprs[2]).with_columns(exprs[3]).collect())
def plugin_1row_eager_from_dict():
    df = pl.DataFrame([record], schema=frame1.schema)
    for e in exprs: df = df.with_columns(e)
    return df.row(0, named=True)
r_plugin_from_dict = bench(plugin_1row_eager_from_dict)
one_tree_expr = dt.walk_value(*[pl.col(f) for f in trees.T1_FEATURES], tree=trees.T1)
r_one_plugin_expr = bench(lambda: frame1.with_columns(one_tree_expr))
r_noop = bench(lambda: frame1.with_columns(dt.noop(pl.col("income"))))
noop_expr = dt.noop(pl.col("income"))
r_noop_prebuilt = bench(lambda: frame1.with_columns(noop_expr))
r_polars_arith_only = bench(lambda: frame1.with_columns((pl.col("income") * 2.0).alias("y")))
for k, v in {"decider2_score_dict": r_score, "decider2_apply_1row": r_apply1,
             "plugin_4exprs_1row_eager": r_plugin_eager, "plugin_4exprs_1row_lazy": r_plugin_lazy,
             "plugin_4exprs_dict_roundtrip": r_plugin_from_dict, "plugin_1expr_1row": r_one_plugin_expr,
             "plugin_noop_1row": r_noop, "plugin_noop_prebuilt_expr_1row": r_noop_prebuilt,
             "polars_native_arith_1row": r_polars_arith_only}.items():
    print(f"{k:34s} p50 {v['p50_us']:9.1f} us   min {v['min_us']:9.1f}   p99 {v['p99_us']:9.1f}")
    log("q4_single_record", path=k, **v)

# ---------------------------------------------------------------- (b) equivalence
for seed in (1, 2, 3):
    df = trees.make_frame(50_000, seed=seed)
    # sprinkle nulls into every feature: the null rung of the equivalence
    df = df.with_columns([pl.when(pl.int_range(pl.len()) % 97 == i).then(None).otherwise(pl.col(c)).alias(c)
                          for i, c in enumerate(df.columns)])
    a = trees.run_polars(df, how="eager")
    b = trees.run_polars(df, how="eager", oracle=True)
    c = trees.run_polars(df, how="streaming", parallel=True)
    for col in ("pts1", "adj", "pts2", "approve"):
        assert a[col].equals(b[col]), (col, "plugin != oracle")
        assert a[col].equals(c[col]), (col, "plugin eager != plugin streaming+parallel")
    nulls = a["pts1"].null_count()
    print(f"seed {seed}: plugin == when/then oracle == plugin(streaming,parallel) on 50k rows incl. {nulls} null paths")
# decider2 handles nulls by policy, so compare it on the all-present frame, and check its own three rungs
df = trees.make_frame(50_000, seed=11)
d2 = {m: p.apply(df, mode=m) for m in ("interpreted", "stepped", "fused")}
pf = trees.run_polars(df)
for col in ("pts1", "pts2", "approve"):
    for m in d2:
        assert d2[m][col].equals(pf[col]), (m, col)
for m in d2:
    assert float((d2[m]["adj"] - pf["adj"]).abs().max()) < 1e-9, (m, "adj")   # numba vs polars rounding, see q4_numeric_divergence
print("decider2 interpreted == stepped == fused == plugin, 50k rows")
log("q4_equivalence", plugin_vs_oracle="equal (3 seeds x 50k, with nulls)", decider2_rungs_vs_plugin="equal (50k)")

# ---------------------------------------------------------------- (c) cost of getting the tree across
def chain(depth):
    """A one-sided chain of `depth` f64 tests, then a leaf: tree size scales linearly."""
    # node i: income < -i (never true) ? leaf 0 : next node; the last test's else is leaf 1
    return [dt.test(0, "<", float(-i), depth, i + 1 if i + 1 < depth else depth + 1) for i in range(depth)] \
        + [dt.leaf(0, 1.0), dt.leaf(1, 2.0)]
frame_1k = trees.make_frame(1_000)
for depth in (8, 64, 512, 4096):
    t = chain(depth)
    kw = {"nodes": t, "parallel": False}
    pk = bench(lambda: pickle.dumps(kw, protocol=5), reps=500)
    build = bench(lambda: dt.walk(pl.col("income"), tree=t), reps=500)          # Expr construction (pickles kwargs)
    e = dt.walk(pl.col("income"), tree=t)
    ev1 = bench(lambda: frame1.with_columns(e), reps=500)                       # eval on 1 row: deserialise + compile + walk 1
    ev1k = bench(lambda: frame_1k.with_columns(e), reps=200)
    nbytes = len(pickle.dumps(kw, protocol=5))
    print(f"depth {depth:5d}  pickled {nbytes:8d} B  pickle {pk['p50_us']:7.1f} us  Expr build {build['p50_us']:7.1f} us  "
          f"eval 1 row {ev1['p50_us']:7.1f} us  eval 1k rows {ev1k['p50_us']:7.1f} us")
    log("q4_tree_transfer", depth=depth, pickled_bytes=nbytes, pickle_p50_us=pk["p50_us"], expr_build_p50_us=build["p50_us"],
        eval_1row_p50_us=ev1["p50_us"], eval_1k_p50_us=ev1k["p50_us"])
# (retune + panic demos live in q4b_retune_panic.py)
