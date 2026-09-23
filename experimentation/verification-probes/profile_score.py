"""Independent check: where does decider2's single-record score() actually
spend its time? The design strand claims ~85% is schema-invariant Python redone
on every call (inspect.signature+eval, a difflib interface walk,
flatten_for_runtime), and only ~1.4 us is the kernel. That would mean the 90%
single-record case is fixed by caching Python, not by Arrow or Rust.
"""
import cProfile, io, pstats, sys, time
sys.path.insert(0, "decider2/src")
sys.path.insert(0, "decider2")
import numpy as np, polars as pl
from decider2 import flow, step
from decider2.trees import (LeafNode, MultiEdgeData, MultiSourceEdge, PositionedNode,
                            Tree, TreeOutput, UnaryGreaterThan, UnaryNode, tree_module)

def one_tree(name, feature):
    return Tree(name=name,
        nodes=[PositionedNode(id="root", data=UnaryNode(condition=UnaryGreaterThan(feature=feature, threshold=0.5))),
               PositionedNode(id="yes", data=LeafNode(result_idx=1)),
               PositionedNode(id="no", data=LeafNode(result_idx=0))],
        edges=[MultiSourceEdge(source="root", target="yes", data=MultiEdgeData(sourceIndex=[0])),
               MultiSourceEdge(source="root", target="no", data=MultiEdgeData(sourceIndex=[1]))],
        output=TreeOutput(data=[{f"{name}_hit": 0}, {f"{name}_hit": 1}],
                          default={f"{name}_hit": -1}, dtypes=[(f"{name}_hit", "Int64")]))

built = tree_module(one_tree("t1", "x"))
pipeline = flow(built.module)
row = {"x": 0.9}
pipeline.score(row)          # warm everything

N = 300
t0 = time.perf_counter()
for _ in range(N):
    pipeline.score(row)
us = (time.perf_counter() - t0) / N * 1e6
print(f"score() single record: {us:.1f} us/call   (spec 60 us)")

pr = cProfile.Profile(); pr.enable()
for _ in range(N):
    pipeline.score(row)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
out = s.getvalue()
print("\n".join(out.splitlines()[4:34]))

# how much is signature/eval/difflib/flatten?
st = pstats.Stats(pr)
buckets = {"inspect.signature / typing eval": 0.0, "difflib": 0.0, "flatten_for_runtime": 0.0}
total = 0.0
for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
    total += tt
    low = f"{fn}:{name}".lower()
    if "inspect" in low or "typing" in low or name == "eval" or "get_type_hints" in low:
        buckets["inspect.signature / typing eval"] += tt
    elif "difflib" in low:
        buckets["difflib"] += tt
    elif "flatten_for_runtime" in name:
        buckets["flatten_for_runtime"] += tt
print(f"\n--- self-time shares of {total*1e6/N:.0f} us/call profiled ---")
for k, v in buckets.items():
    print(f"  {k:34s} {v/total*100:5.1f}%   ({v*1e6/N:6.1f} us/call)")
