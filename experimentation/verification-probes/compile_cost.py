"""What does inline="always" cost at BUILD time, and does it still disk-cache?

Same tree as bench_typed_features.py's `mixed`. Prints cold compile wall time
and counts numba's own cache save/load lines.
"""
import os, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, polars as pl
import decider2
from decider2 import flow
from decider2.trees import tree_module
from bench import build_tree, build_frame

label = sys.argv[1]
tree, feature_types = build_tree("mixed")
frame = build_frame("mixed", 20_000)
t0 = time.perf_counter()
try:
    built = tree_module(tree, feature_types=feature_types)
except TypeError:
    built = tree_module(tree)
t_build = time.perf_counter() - t0
pipeline = flow(built.module)
t0 = time.perf_counter()
out = pipeline.apply(frame, mode="fused")
t_first = time.perf_counter() - t0
t0 = time.perf_counter()
pipeline.apply(frame, mode="fused")
t_second = time.perf_counter() - t0
print(json.dumps({
    "label": label,
    "decider2_file": str(Path(decider2.__file__).resolve()),
    "build_s": round(t_build, 3),
    "first_apply_s": round(t_first, 3),
    "second_apply_s": round(t_second, 4),
}))
