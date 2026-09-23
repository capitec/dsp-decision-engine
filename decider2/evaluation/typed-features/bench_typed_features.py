"""ns/row of a realistic decision tree, before vs after typed features.

Run the SAME script against the unmodified tree (`--label before`) and
against the typed-feature tree (`--label after`); it appends one JSON line
per measurement to `measurements.jsonl` next to it, so a crash loses
nothing already measured.

    PYTHONPATH=src <python> evaluation/typed-features/bench_typed_features.py --label before

Two trees, both ~40 leaves, 16 features, depth ~7, mixed ops
(`<`/`>=`/`==`/`between`/`is_true`), one seed, so before and after walk the
identical structure over the identical 200k-row frame:

  single : every feature Float64 — the shape the single float64 array was
           built for; the typed path can only lose here (extra `feat_kind`
           load + switch per node, four gather loops instead of one).
  mixed  : 10 Float64 + 4 Int64 (scaled cents, counts) + 2 Boolean — the
           shape doc 03 §1.2 says real money columns take. Before: the
           Int64/Boolean columns are cast to float64 at the boundary.
           After: they stay int64/bool end to end.

Two numbers per tree:

  apply_ns_per_row : `pipeline.apply(frame, mode="fused")` end to end —
                     boundary extraction, the path kernel, the output-column
                     kernel, write-back. What a caller actually pays.
  path_ns_per_row  : the `<tree>_path` packed segment's `.run()` alone —
                     row gather + walk, nothing else. Where the typed
                     representation's cost actually lands.

Median of `--reps` timed runs after two warm-ups; each run is the whole
frame, so one rep is ~200k rows and timer resolution is not a factor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
OUT = HERE / "measurements.jsonl"

# --- the realistic tree -----------------------------------------------------

FLOAT_FEATURES = {
    "credit_score": (300.0, 850.0),
    "dti_ratio": (0.0, 1.0),
    "loan_to_value": (0.0, 1.5),
    "utilisation": (0.0, 1.0),
    "age": (18.0, 90.0),
    "months_on_book": (0.0, 240.0),
    "months_employed": (0.0, 480.0),
    "affordability_ratio": (0.0, 2.0),
    "enquiries_12m": (0.0, 20.0),
    "bureau_score": (0.0, 1000.0),
}
INT_FEATURES = {  # scaled int64 cents / counts — doc 03 §1.2
    "monthly_income_cents": (0, 5_000_000_00),
    "instalment_cents": (0, 500_000_00),
    "existing_defaults": (0, 5),
    "open_accounts": (0, 30),
}
BOOL_FEATURES = ["is_staff", "has_arrears"]


def build_tree(variant: str, *, target_leaves: int = 40, seed: int = 7):
    """`variant` is "single" (all 16 features Float64) or "mixed"."""
    from decider2.trees import (
        LeafNode, MultiEdgeData, MultiSourceEdge, PositionedNode, Tree, TreeOutput,
        UnaryBetween, UnaryEqual, UnaryGreaterThanEqual, UnaryIsTrue, UnaryLessThan, UnaryNode,
    )

    rng = random.Random(seed)
    feats: list[tuple[str, str, tuple]] = []  # (name, kind, range)
    for name, r in FLOAT_FEATURES.items():
        feats.append((name, "float", r))
    for name, r in INT_FEATURES.items():
        feats.append((name, "int" if variant == "mixed" else "float", r))
    for name in BOOL_FEATURES:
        feats.append((name, "bool" if variant == "mixed" else "float", (0, 1)))

    nodes: list = []
    edges: list = []
    leaf_rows: list = []
    counter = [0]

    def nid() -> str:
        counter[0] += 1
        return f"n{counter[0]}"

    def build(budget: int) -> str:
        my = nid()
        if budget <= 1:
            idx = len(leaf_rows)
            leaf_rows.append({"score": float(idx), "band": idx % 5})
            nodes.append(PositionedNode(id=my, data=LeafNode(result_idx=idx)))
            return my
        name, kind, (lo, hi) = rng.choice(feats)
        if kind == "bool" or (variant == "single" and name in BOOL_FEATURES):
            cond = UnaryIsTrue(feature=name)
        elif kind == "int":
            t = rng.randint(lo, hi)
            cond = rng.choice([
                lambda: UnaryLessThan(feature=name, threshold=t),
                lambda: UnaryGreaterThanEqual(feature=name, threshold=t),
                lambda: UnaryEqual(feature=name, threshold=rng.randint(lo, min(hi, lo + 3))),
                lambda: UnaryBetween(feature=name, min=lo + (hi - lo) // 4, max=lo + 3 * (hi - lo) // 4),
            ])()
        else:
            t = rng.uniform(lo, hi) if isinstance(lo, float) else float(rng.randint(lo, hi))
            lo_f, hi_f = float(lo), float(hi)
            cond = rng.choice([
                lambda: UnaryLessThan(feature=name, threshold=t),
                lambda: UnaryGreaterThanEqual(feature=name, threshold=t),
                lambda: UnaryBetween(feature=name, min=lo_f + (hi_f - lo_f) / 4, max=lo_f + 3 * (hi_f - lo_f) / 4),
            ])()
        nodes.append(PositionedNode(id=my, data=UnaryNode(condition=cond)))
        left = build(budget // 2)
        right = build(budget - budget // 2)
        edges.append(MultiSourceEdge(source=my, target=left, data=MultiEdgeData(sourceIndex=[0])))
        edges.append(MultiSourceEdge(source=my, target=right, data=MultiEdgeData(sourceIndex=[1])))
        return my

    build(target_leaves)
    tree = Tree(
        name=f"credit_{variant}",
        nodes=nodes, edges=edges,
        output=TreeOutput(
            data=leaf_rows, default={"score": -1.0, "band": -1},
            dtypes=[("score", "Float64"), ("band", "Int64")],
        ),
    )
    read = tree.required_features()
    feature_types = {name: kind for name, kind, _ in feats if name in read}
    return tree, feature_types


def build_frame(variant: str, n: int, *, seed: int = 11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols: dict = {}
    for name, (lo, hi) in FLOAT_FEATURES.items():
        cols[name] = pl.Series(name, rng.uniform(lo, hi, n), dtype=pl.Float64)
    for name, (lo, hi) in INT_FEATURES.items():
        vals = rng.integers(lo, hi + 1, n, dtype=np.int64)
        cols[name] = pl.Series(name, vals, dtype=pl.Int64 if variant == "mixed" else pl.Float64)
    for name in BOOL_FEATURES:
        vals = rng.random(n) < 0.3
        cols[name] = pl.Series(name, vals, dtype=pl.Boolean if variant == "mixed" else pl.Float64)
    return pl.DataFrame(cols)


# --- timing -----------------------------------------------------------------


def _median_ns_per_row(fn, n_rows: int, reps: int) -> tuple[float, float]:
    for _ in range(2):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / n_rows)
    return statistics.median(samples), min(samples)


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def record(row: dict) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="before | after")
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--variants", default="single,mixed")
    args = ap.parse_args()

    import decider2
    from decider2 import flow
    from decider2.compile.driver import _DRIVER_CACHE, PackedCompiledSegment, numpy_dtype
    from decider2.runtime.invoke import resolve_params
    from decider2.trees import tree_module

    src = str(Path(decider2.__file__).resolve())
    print("decider2:", src)
    base = {
        "label": args.label, "git": _git_rev(), "decider2_file": src,
        "rows": args.rows, "reps": args.reps,
        "python": platform.python_version(), "machine": platform.node(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for variant in args.variants.split(","):
        tree, feature_types = build_tree(variant)
        frame = build_frame(variant, args.rows)
        typed_api = False
        try:
            built = tree_module(tree, feature_types=feature_types)
            typed_api = True
        except TypeError:
            built = tree_module(tree)
        pipeline = flow(built.module)
        n = frame.height

        # -- end to end -------------------------------------------------
        out = pipeline.apply(frame, mode="fused")
        digest = hashlib.sha256(out[built.path_column].to_numpy().tobytes()).hexdigest()[:16]
        med, best = _median_ns_per_row(lambda: pipeline.apply(frame, mode="fused"), n, args.reps)
        row = {**base, "tree": variant, "typed_api": typed_api, "leaves": built.encoded.leaf_count,
               "max_depth": built.encoded.max_depth, "features": len(built.encoded.features),
               "path_digest": digest, "metric": "apply_ns_per_row", "median": med, "min": best}
        record(row)
        print(f"{args.label:6s} {variant:6s} apply : median {med:8.1f} ns/row  min {best:8.1f}  digest {digest}")

        # -- the path segment alone --------------------------------------
        steps, group_ids, owners, param_spaces = pipeline.flatten_for_runtime()
        driver = None
        for d in _DRIVER_CACHE.values():
            if any(s.name == built.path_column for seg in d.segments for s in seg.steps):
                driver = d
        assert driver is not None
        seg = next(s for s in driver.segments if isinstance(s, PackedCompiledSegment) and s.steps[0].name == built.path_column)
        by_name = {i.name: i for i in pipeline.interface.inputs}
        registry = {}
        for inp in seg.steps[0].inputs:
            col = frame[inp.name].to_numpy()
            registry[inp.name] = col.astype(numpy_dtype(by_name[inp.name].annotation), copy=False)
        resolved = resolve_params(steps, None, param_spaces=param_spaces, owners=owners)
        input_dtypes = sorted({str(v.dtype) for v in registry.values()})
        med, best = _median_ns_per_row(lambda: seg.run(registry, resolved, n), n, args.reps * 2)
        row = {**base, "tree": variant, "typed_api": typed_api, "leaves": built.encoded.leaf_count,
               "max_depth": built.encoded.max_depth, "features": len(built.encoded.features),
               "path_digest": digest, "metric": "path_ns_per_row", "median": med, "min": best}
        record(row)
        print(f"{args.label:6s} {variant:6s} path  : median {med:8.1f} ns/row  min {best:8.1f}")
        # the registry dtypes the path step actually saw
        print("        path-step input dtypes:", input_dtypes)


if __name__ == "__main__":
    main()
