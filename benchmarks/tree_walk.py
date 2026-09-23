"""Tree walk throughput and score() latency of a TreeConfig in every mode.

The tree is a campaign-style v3 document: 63 internal nodes over 16 float and
2 int features (thresholds, AND composites, between), 64 leaves, and a
String, a Float64 and an Int64 output. The second run puts a string match on
a `channel` column (starts_with, byte spans in the kernel) in front of it.

    uv run python benchmarks/tree_walk.py
"""
import gc
import time

import numpy as np
import polars as pl

from decider.engine import Engine
from decider.steps.trees import TreeConfig

N = 1_000_000
FLOATS = [f"f{i}" for i in range(16)]
INTS = ["months", "enquiries"]
rng = np.random.default_rng(0)


def _document(depth: int = 6) -> dict:
    nodes, edges, leaves = [], [], []

    def feature():
        return str(rng.choice(FLOATS + INTS))

    def threshold(f):
        return int(rng.integers(0, 24)) if f in INTS else round(float(rng.uniform(0.2, 0.8)), 3)

    def grow(d: int) -> str:
        nid = f"n{len(nodes)}"
        if d == depth:
            nodes.append({"id": nid, "data": {"type": "leaf", "result_idx": len(leaves)}})
            leaves.append(len(leaves))
            return nid
        shape = rng.integers(3)
        if shape == 0:
            f = feature()
            data = {"type": "unary", "condition": {"op": ">=", "feature": f, "threshold": threshold(f)}}
        elif shape == 1:
            conds = []
            for _ in range(int(rng.integers(2, 4))):
                f = feature()
                conds.append({"op": "<", "feature": f, "threshold": threshold(f)})
            data = {"type": "composite", "op": "and", "conditions": conds}
        else:
            f = str(rng.choice(FLOATS))
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


def _gated(doc: dict) -> dict:
    gate = {"id": "gate", "data": {"type": "unary", "condition": {
        "op": "string_match", "feature": "channel", "patterns": ["app", "web"], "match_type": "starts_with"}}}
    edge = {"source": "gate", "target": "n0", "data": {"sourceIndex": [0]}}
    return {**doc, "nodes": [gate, *doc["nodes"]], "edges": [edge, *doc["edges"]]}


DOC = _document()
TYPES = {f: "int" for f in INTS}
frame = pl.DataFrame({**{f: rng.uniform(0, 1, N) for f in FLOATS},
                      **{f: rng.integers(0, 24, N) for f in INTS},
                      "channel": rng.choice(["app-ios", "web", "branch", "call centre"], N)})
ROW = frame.row(0, named=True)


def _time(f, *args):
    t = time.perf_counter()
    f(*args)
    return time.perf_counter() - t


def rows_per_s(run, reps=7):
    run(frame)
    return N / min(_time(run, frame) for _ in range(reps))


def latency(score, calls=20000):
    for _ in range(1000):
        score(ROW)
    gc.collect()
    samples = sorted(_time(score, ROW) for _ in range(calls))
    return samples[len(samples) // 2] * 1e6, samples[int(len(samples) * 0.99)] * 1e6


def compare(doc: dict) -> None:
    tree = TreeConfig(name="campaign", tree=doc, feature_types=TYPES)
    engines = {}
    for mode in ("fused", "stepped", "interpreted"):
        exe = Engine().bind(tree, mode=mode)
        engines[f"decider {mode}"] = (exe.run, exe.score)

    # The outputs agree before anything is timed.
    expected = engines["decider interpreted"][0](frame.head(20000)).select("label", "score", "band")
    for name, (run, _) in engines.items():
        assert run(frame.head(20000)).select("label", "score", "band").equals(expected), name

    latencies = {name: latency(score, calls=2000 if "interpreted" in name else 20000)
                 for name, (_, score) in engines.items()}
    results = {name: (rows_per_s(run, reps=1 if "interpreted" in name else 7), *latencies[name])
               for name, (run, _) in engines.items()}
    print(f"{'engine':<22}{'batch rows/s':>16}{'ns/row':>10}{'score p50 us':>14}{'score p99 us':>14}")
    for name, (batch, p50, p99) in results.items():
        print(f"{name:<22}{batch:>16,.0f}{1e9 / batch:>10.1f}{p50:>14.1f}{p99:>14.1f}")


print("numeric tree")
compare(DOC)
print("gated by a string match")
compare(_gated(DOC))
