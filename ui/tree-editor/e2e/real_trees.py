"""Every tree document in the repo, plus generated flat-rule sets: the one list the real-tree tests run over.

Each `RealTree` holds its `TreeConfig` step document (`{"type": "tree", "name": ..., <step settings>, "tree": doc}`);
`document` is the tree document the editor imports.
"""

import contextlib
import functools
import importlib.util
import io
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from decikit import TreeConfig, flow
from decikit.steps.trees.schema import detect_format

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RealTree:
    id: str
    origin: str
    step: dict
    file: Path | None = None  # the JSON file as it is stored, when the tree is one

    @property
    def document(self) -> dict:
        return self.step["tree"]

    @property
    def flat(self) -> bool:
        return detect_format(self.document) != "v3"

    @property
    def settings(self) -> dict:
        """The step settings around the document (`feature_types`, `path_output`, ...), which the editor never sees."""
        return {k: v for k, v in self.step.items() if k != "tree"}

    def wrap(self, document: dict) -> dict:
        """`document` as this tree's step document, so an export runs with the same step settings."""
        return {**self.settings, "tree": document}


def tree_files() -> set[Path]:
    """Every JSON file in the repo `TreeConfig.load` accepts as it is or wrapped (dependencies and builds aside)."""
    found = set()
    for folder, subfolders, names in os.walk(REPO):
        subfolders[:] = [d for d in subfolders if d not in ("node_modules", "build", "dist") and not d.startswith(".")]
        for path in (Path(folder, n) for n in names if n.endswith(".json")):
            with contextlib.suppress(Exception):
                doc = json.loads(path.read_text())
                TreeConfig.load(doc if doc.get("type") == "tree" else _wrapped("t", doc))
                found.add(path)
    return found


def _wrapped(name: str, document: dict) -> dict:
    return {"type": "tree", "name": name, "tree": document}


def _files() -> list[RealTree]:
    files = {
        "campaign-23": "examples/campaign-trees/configs/0.1.0/campaign_23.json",
        "mcp-flat-rules": "tests/mcp/data/flat_rules.json",
        "bank-bureau-grade": "ui/vscode/examples/bank/trees/bureau_grade.json",
        "bank-collections-risk": "ui/vscode/examples/bank/trees/collections_risk.json",
        "bank-fraud-screen": "ui/vscode/examples/bank/trees/fraud_screen.json",
    }
    trees = [RealTree(i, p, json.loads((REPO / p).read_text()), REPO / p) for i, p in files.items()]
    stored = "ui/tree-editor/backend/tests/data/stored_documents.json"
    for key, document in json.loads((REPO / stored).read_text()).items():
        trees.append(RealTree(f"stored-{key.replace('_', '-')}", f"{stored} [{key}]", _wrapped(key, document)))
    return trees


@contextlib.contextmanager
def _loads():
    """Every dict `TreeConfig.load` accepts inside the block, as it was passed."""
    seen, load = [], TreeConfig.load.__func__

    def capture(cls, doc, *args, **kwargs):
        step = load(cls, doc, *args, **kwargs)
        if isinstance(doc, dict):
            seen.append(doc)
        return step

    TreeConfig.load = classmethod(capture)
    try:
        yield seen
    finally:
        TreeConfig.load = classmethod(load)


def _docs() -> list[RealTree]:
    # A sample is the step document a published example passes to `TreeConfig.load`: JSON blocks as written (a bare
    # tree document wrapped), Python blocks run with the names the docs leave implied. A block that fails keeps what
    # it loaded first.
    pages = [REPO / "decikit/GUIDE.md", *sorted((REPO / "docs").rglob("*.md"))]
    found: dict[str, tuple[str, dict]] = {}
    for page in pages:
        text, where = page.read_text(), str(page.relative_to(REPO))
        for block in re.findall(r"```json\n(.*?)```", text, re.S):
            with contextlib.suppress(Exception):
                doc = json.loads(block)
                doc = doc if doc.get("type") == "tree" else _wrapped(doc.get("name") or "tree", doc)
                TreeConfig.load(doc)
                found.setdefault(json.dumps(doc, sort_keys=True), (where, doc))
        for block in (b for b in re.findall(r"```python\n(.*?)```", text, re.S) if "TreeConfig.load(" in b):
            with _loads() as seen, contextlib.redirect_stdout(io.StringIO()), contextlib.suppress(Exception):
                exec(compile(block, where, "exec"), {"__name__": "docs_sample", "TreeConfig": TreeConfig, "pl": pl})
            for doc in seen:
                found.setdefault(json.dumps(doc, sort_keys=True), (where, doc))
    trees, seen_ids = [], {}
    for where, doc in found.values():
        base = f"docs-{Path(where).stem.lower()}-{doc['name']}".replace("_", "-")
        seen_ids[base] = seen_ids.get(base, 0) + 1
        trees.append(RealTree(base if seen_ids[base] == 1 else f"{base}-{seen_ids[base]}", where, doc))
    return trees


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _campaign_scale() -> list[RealTree]:
    # The campaign example's scale test builds a tree the size its analysts publish (hundreds of nodes) in Python.
    sys.path.insert(0, str(REPO / "examples/campaign-trees"))
    source = "examples/campaign-trees/tests/test_scale.py"
    return [RealTree("campaign-scale", f"{source} random_tree(150, seed=7)",
                     _module("campaign_scale", source).random_tree(150, seed=7))]


def _fraud_example() -> list[RealTree]:
    # The fraud example's rule set is its own step type; its live, shadow and overlay trees exist only in Python.
    sys.path.insert(0, str(REPO / "examples/transaction-fraud"))
    from fraud_interdiction.rules import RuleSet

    source = "examples/transaction-fraud/configs/0.1.0/rule_set.json"
    with _loads() as seen:
        flow(RuleSet.load(str(REPO / source)), name="fraud").parameters()
    return [RealTree(f"fraud-{doc['name'].replace('_', '-')}", f"{source} via fraud_interdiction/rules.py", doc)
            for doc in seen]


GENERATED_SIZES = (1, 32, 64, 65, 200, 521)
# A tree with params is never split into parts, so the sizes just past the 64-output part keep literal thresholds.
SPLIT_SIZES = (65, 521)


@functools.cache
def _sweep():
    return _module("tree_split_sweep", "benchmarks/tree_split_sweep.py")


def generated(n: int) -> dict:
    """A `mode: "all"` step document of `n` rules in the shapes of benchmarks/tree_split_sweep.py.

    Rule R0 is `f0 > r0_limit`, a param, except in the SPLIT_SIZES sets.
    """
    sweep = _sweep()
    rng, features = random.Random(n), sweep._features(sweep.SHAPES["fraud"][0])
    shapes = ["fraud", "fraud", "fraud", "isin_num", "or", "between", "threshold"]
    with_param = n not in SPLIT_SIZES
    first = {"op": ">", "feature": "f0", "threshold": {"param": "r0_limit"} if with_param else 5}
    rules = [{"meta": {"name": "R0"}, "rule": {"type": "unary", "condition": first, "then": sweep.LEAF}}]
    rules += [{"meta": {"name": f"R{i}"}, "rule": sweep.SHAPES[shapes[i % len(shapes)]][1](rng, features)}
              for i in range(1, n)]
    document = {"type": "prioritized_flat_rule", "mode": "all", "rules": rules, "output": sweep.OUTPUT}
    if with_param:
        document["parameters"] = {"r0_limit": {"type": "Float64", "default_value": 5.0}}
    step = {"type": "tree", "name": f"rules_{n}", "tree": document}
    # As the benchmark declares it; a feature_types entry for a feature no rule reads is an error.
    return {**step, "feature_types": {"flag": "bool"}} if '"flag"' in json.dumps(rules) else step


def _generated() -> list[RealTree]:
    return [RealTree(f"generated-{n}", "real_trees.generated", generated(n)) for n in GENERATED_SIZES]


# Documents the editor could import that the tests leave out, and why.
EXCLUDED = {
    "stored-default-flat": "decikit refuses to build it: an empty rule set whose output declares no columns",
    "docs-guide-band": "only loaded when z3 (the `prove` extra) is installed, so the e2e environment never has it",
    "docs-guide-band-2": "as docs-guide-band",
}
TREES: list[RealTree] = [t for t in (*_files(), *_docs(), *_campaign_scale(), *_fraud_example(), *_generated())
                         if t.id not in EXCLUDED]
BY_ID = {tree.id: tree for tree in TREES}
assert len(BY_ID) == len(TREES), "tree ids must be unique"
