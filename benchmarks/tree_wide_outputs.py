"""Compile time of a `mode: "all"` tree against its rule count.

`uv run python benchmarks/tree_wide_outputs.py 250` builds a tree of 250 similar rules (three conditions each,
one boolean output per rule) and times `bind` plus the first `score`. Set `NUMBA_CACHE_DIR` to a fresh directory
for a cold run and run it twice for a warm one. `--whole` keeps the tree as one walker call, as it was before
trees were split.
"""
import random
import sys
import time

from decider import TreeConfig, flow
from decider._engine import Engine
from decider.steps.trees import _split


def tree(n: int) -> TreeConfig:
    rng = random.Random(1)
    features = [f"f{i}" for i in range(12)]
    leaf = {"type": "leaf", "result_idx": 0}
    rules = []
    for k in range(n):
        conditions = [{"op": ">=", "feature": rng.choice(features), "threshold": rng.randint(1, 9)},
                      {"op": ">", "feature": rng.choice(features), "threshold": rng.randint(1, 9)},
                      {"op": "between", "feature": rng.choice(features), "min": 1, "max": rng.randint(2, 9)}]
        rules.append({"meta": {"name": f"R{k}"},
                      "rule": {"type": "composite", "op": "and", "conditions": conditions, "then": leaf}})
    return TreeConfig.load({"type": "tree", "name": "rules", "tree": {
        "type": "prioritized_flat_rule", "mode": "all", "rules": rules,
        "output": {"data": [{"fired": True}], "default": {"fired": False}, "dtypes": [["fired", "Boolean"]]}}})


if __name__ == "__main__":
    n = int(sys.argv[1])
    if "--whole" in sys.argv:
        _split.MAX_OUTPUTS = 10**9
    start = time.time()
    Engine().bind(flow(tree(n), name="p"), mode="fused").score({f"f{i}": float(i) for i in range(12)})
    print(f"{n} rules: {time.time() - start:.1f} s")
