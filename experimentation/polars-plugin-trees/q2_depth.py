"""Depth scaling, 1M rows: row-walk plugin vs pure-polars when/then (evaluates every condition) vs decider2 fused,
on full binary trees of depth 3/5/7/9 over f64 features. Where does a row walk beat the columnar engine?"""
import gc, sys, time, polars as pl, numpy as np
from common import log
import decider_trees as dt, trees
sys.path.insert(0, "/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/src")
from decider2 import flow
from decider2.trees import tree_module

N = 1_000_000
rng = np.random.default_rng(3)
NF = 6
df = pl.DataFrame({f"x{i}": rng.random(N) for i in range(NF)})
feats = [f"x{i}" for i in range(NF)]

def full_binary(depth):
    """node k tests feature k%NF against a random threshold; leaves numbered 0..2^depth-1."""
    nodes, leaf_n = [], [0]
    def build(d):
        k = len(nodes); nodes.append(None)
        if d == depth:
            nodes[k] = dt.leaf(leaf_n[0], float(leaf_n[0])); leaf_n[0] += 1; return k
        thr = float(rng.random())
        t = build(d + 1); e = build(d + 1)
        nodes[k] = dt.test(k % NF, "<", thr, t, e); return k
    build(0); return nodes

def timeit(fn, reps=5):
    fn(); ts = []
    for _ in range(reps):
        gc.collect(); t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts)//2]

cols = [pl.col(f) for f in feats]
for depth in (3, 5, 7, 9):
    tree = full_binary(depth)
    e_plugin = dt.walk_value(*cols, tree=tree).alias("v")
    e_par = dt.walk_value(*cols, tree=tree, parallel=True).alias("v")
    e_oracle = dt.to_when_then(tree, cols, value=True).alias("v")
    a = df.with_columns(e_plugin)["v"]; b = df.with_columns(e_oracle)["v"]
    assert a.equals(b)
    d2 = flow(tree_module(trees.to_decider2_tree(tree, feats, f"fb{depth}", "v")).module)
    d2.precompile()
    c = d2.apply(df, mode="fused")["v"]; assert c.equals(a)
    row = {"depth": depth, "leaves": 2 ** depth, "threads": pl.thread_pool_size()}
    row["plugin_ns"] = timeit(lambda: df.with_columns(e_plugin)) / N * 1e9
    row["plugin_par_ns"] = timeit(lambda: df.with_columns(e_par)) / N * 1e9
    row["when_then_ns"] = timeit(lambda: df.with_columns(e_oracle)) / N * 1e9
    row["when_then_streaming_ns"] = timeit(lambda: df.lazy().with_columns(e_oracle).collect(engine="streaming")) / N * 1e9
    row["decider2_fused_ns"] = timeit(lambda: d2.apply(df, mode="fused")) / N * 1e9
    row["loadavg_1m"] = float(open("/proc/loadavg").read().split()[0])
    print(f"depth {depth} ({2**depth:3d} leaves): plugin {row['plugin_ns']:6.1f}  plugin-par {row['plugin_par_ns']:6.1f}  "
          f"when/then {row['when_then_ns']:6.1f}  when/then-streaming {row['when_then_streaming_ns']:6.1f}  decider2 {row['decider2_fused_ns']:6.1f} ns/row  load {row['loadavg_1m']:.1f}")
    log("q2_depth_1M", **row)
