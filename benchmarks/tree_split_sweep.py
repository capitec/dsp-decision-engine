"""Part size of a split `mode: "all"` tree against compile time, single-record latency and batch speed.

    uv run python benchmarks/tree_split_sweep.py sweep --shapes and3,fraud --counts 100,521 --out .scratch/sweep.jsonl
    uv run python benchmarks/tree_split_sweep.py one and3 250 32      # one cell in this process

`sweep` runs every cell in two fresh processes sharing one `NUMBA_CACHE_DIR`: the first is cold (bind plus the
first record `run` compile everything), the second is warm (the disk cache is hot), and both then time `run(record)` and a
batch `run()`. Part sizes are in outputs; a rule writes one boolean, so that is rules per part. `whole` is unsplit.
Run the sweep under one wrapped process so other work cannot compete for the cores.
"""
import argparse
import gc
import json
import os
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

SIZES = ("16", "32", "64", "128", "256", "whole")
LEAF = {"type": "leaf", "result_idx": 0}
OUTPUT = {"data": [{"fired": True}], "default": {"fired": False}, "dtypes": [["fired", "Boolean"]]}
WORDS = [f"w{i}" for i in range(40)]


def _features(k: int) -> list[str]:
    return [f"f{i}" for i in range(k)]


def _cmp(rng, feats, op=None):
    return {"op": op or rng.choice([">", ">=", "<"]), "feature": rng.choice(feats), "threshold": rng.randint(1, 9)}


def _between(rng, feats):
    return {"op": "between", "feature": rng.choice(feats), "min": 1, "max": rng.randint(2, 9)}


def _and(*conds):
    return {"type": "composite", "op": "and", "conditions": list(conds), "then": LEAF}


def _rule(root):
    return root if root.get("type") == "composite" else {"type": "unary", "condition": root, "then": LEAF}


def _fraud(rng, feats):
    # The shape counts of examples/transaction-fraud's rule_set.json: and3 40%, and2 35%, or2 12%, and5 5%, single 8%,
    # over about twenty features with comparisons, bands, flags and a few computed ratios.
    def leaf():
        r = rng.random()
        if r < 0.05:
            return {"op": ">", "feature": {"type": "computed", "expression": f"{feats[0]} - {rng.randint(1, 9)}.0 * {feats[1]}"},
                    "threshold": 0.0}
        return _between(rng, feats) if r < 0.35 else {"op": "is_true", "feature": "flag"} if r < 0.45 else _cmp(rng, feats)

    r = rng.random()
    if r < 0.08:
        return _rule(leaf())
    n = 3 if r < 0.48 else 2 if r < 0.83 else 5 if r > 0.95 else 0
    if n:
        return _and(*[leaf() for _ in range(n)])
    return {"type": "composite", "op": "or", "conditions": [leaf(), leaf()], "then": LEAF}


# shape -> (feature count, rule builder); the features are floats unless the shape says string
SHAPES = {
    "threshold": (12, lambda rng, f: _rule(_cmp(rng, f))),
    "and3": (12, lambda rng, f: _and(_cmp(rng, f), _cmp(rng, f), _between(rng, f))),
    "and3_f4": (4, lambda rng, f: _and(_cmp(rng, f), _cmp(rng, f), _between(rng, f))),
    "and3_f40": (40, lambda rng, f: _and(_cmp(rng, f), _cmp(rng, f), _between(rng, f))),
    "or": (12, lambda rng, f: {"type": "composite", "op": "or", "conditions": [
        _cmp(rng, f), _and(_cmp(rng, f), _cmp(rng, f)), _cmp(rng, f)], "then": LEAF}),
    "between": (12, lambda rng, f: _rule(_between(rng, f))),
    "isin_num": (12, lambda rng, f: _rule({"op": "isin", "feature": rng.choice(f),
                                           "values": [float(rng.randint(0, 9)) for _ in range(4)]})),
    "isin_str": (12, lambda rng, f: _rule({"op": "string_match", "feature": rng.choice(f), "match_type": "exact",
                                           "patterns": rng.sample(WORDS, 4)})),
    "str_eq": (12, lambda rng, f: _rule({"op": "string_match", "feature": rng.choice(f), "match_type": "exact",
                                         "patterns": [rng.choice(WORDS)]})),
    "fraud": (20, _fraud),
}
STRING_SHAPES = {"isin_str", "str_eq"}


def tree(shape: str, n: int, part: str):
    from decikit import TreeConfig

    k, build = SHAPES[shape]
    rng = random.Random(1)
    feats = _features(k)
    rules = [{"meta": {"name": f"R{i}"}, "rule": build(rng, feats)} for i in range(n)]
    config = {"part_outputs": None if part == "whole" else int(part)}
    if shape == "fraud":
        config["feature_types"] = {"flag": "bool"}
    if shape in STRING_SHAPES:
        config["feature_types"] = {f: "str" for f in feats}
    return TreeConfig.load({"type": "tree", "name": "rules", **config, "tree": {
        "type": "prioritized_flat_rule", "mode": "all", "rules": rules, "output": OUTPUT}})


def record(shape: str, rng: random.Random) -> dict:
    k = SHAPES[shape][0]
    if shape in STRING_SHAPES:
        return {f: rng.choice(WORDS) for f in _features(k)}
    row = {f: float(rng.randint(0, 10)) for f in _features(k)}
    return {**row, "flag": True} if shape == "fraud" else row


def frame(shape: str, rows: int):
    import polars as pl

    rng = np.random.default_rng(2)
    k = SHAPES[shape][0]
    if shape in STRING_SHAPES:
        words = np.array(WORDS)
        return pl.DataFrame({f: words[rng.integers(0, len(words), rows)] for f in _features(k)})
    cols = {f: rng.integers(0, 11, rows).astype(float) for f in _features(k)}
    if shape == "fraud":
        cols["flag"] = rng.integers(0, 2, rows).astype(bool)
    return pl.DataFrame(cols)


def one(shape: str, n: int, part: str, rows: int, calls: int) -> dict:
    from decikit import flow
    from decikit._engine import FusedEngine

    warnings.simplefilter("ignore")
    start = time.perf_counter()
    exe = FusedEngine().bind(flow(tree(shape, n, part), name="p"))
    rng = random.Random(3)
    exe.run(record(shape, rng))
    bind = time.perf_counter() - start
    rec = [record(shape, rng) for _ in range(64)]
    for i in range(1000):
        exe.run(rec[i % 64])
    gc.collect()
    samples = np.empty(calls)
    perf = time.perf_counter
    for i in range(calls):
        r = rec[i % 64]
        t = perf()
        exe.run(r)
        samples[i] = perf() - t
    df = frame(shape, rows)
    exe.run(df.head(1000))
    batch = []
    for _ in range(3):
        t = perf()
        exe.run(df)
        batch.append(perf() - t)
    return {"shape": shape, "rules": n, "part": part, "bind_s": round(bind, 2),
            "p50_us": round(float(np.percentile(samples, 50)) * 1e6, 1),
            "p99_us": round(float(np.percentile(samples, 99)) * 1e6, 1),
            "batch_s": round(min(batch), 3), "rows": rows,
            "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}


def cell(shape: str, n: int, part: str, rows: int, calls: int, cache: str) -> dict:
    env = {**os.environ, "NUMBA_CACHE_DIR": cache}
    out = {}
    for phase in ("cold", "warm"):
        proc = subprocess.run([sys.executable, __file__, "one", shape, str(n), part, "--rows", str(rows),
                               "--calls", str(calls)], env=env, capture_output=True, text=True)
        if proc.returncode:
            return {"shape": shape, "rules": n, "part": part, "error": proc.stderr[-400:]}
        out[phase] = json.loads(proc.stdout.strip().splitlines()[-1])
    cold, warm = out["cold"], out["warm"]
    return {"shape": shape, "rules": n, "part": part, "cold_s": cold["bind_s"], "warm_s": warm["bind_s"],
            "p50_us": [cold["p50_us"], warm["p50_us"]], "p99_us": [cold["p99_us"], warm["p99_us"]],
            "batch_s": [cold["batch_s"], warm["batch_s"]], "rows": rows,
            "peak_mb": max(cold["peak_mb"], warm["peak_mb"])}


def sweep(args) -> None:
    counts = [int(c) for c in args.counts.split(",")]
    sizes = args.sizes.split(",")
    root = tempfile.mkdtemp(dir=".scratch", prefix="sweep-")
    for shape in args.shapes.split(","):
        for n in counts:
            # A part of n rules or more is the whole tree: run it once, as `whole`.
            for part in [s for s in sizes if s == "whole" or int(s) < n]:
                if part == "whole" and n > args.whole_max:
                    continue
                cache = os.path.join(root, "cache")
                shutil.rmtree(cache, ignore_errors=True)
                result = cell(shape, n, part, args.rows, args.calls, cache)
                with open(args.out, "a") as f:
                    f.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--shapes", default=",".join(SHAPES))
    s.add_argument("--counts", default="50,100,250,521")
    s.add_argument("--sizes", default=",".join(SIZES))
    s.add_argument("--rows", type=int, default=200_000)
    s.add_argument("--calls", type=int, default=3000)
    s.add_argument("--whole-max", type=int, default=10**9)
    s.add_argument("--out", default=".scratch/sweep.jsonl")
    o = sub.add_parser("one")
    o.add_argument("shape")
    o.add_argument("rules", type=int)
    o.add_argument("part")
    o.add_argument("--rows", type=int, default=200_000)
    o.add_argument("--calls", type=int, default=3000)
    a = ap.parse_args()
    if a.cmd == "sweep":
        sweep(a)
    else:
        print(json.dumps(one(a.shape, a.rules, a.part, a.rows, a.calls)))
