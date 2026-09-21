"""Canonical tree shapes, built ONCE and rendered two ways.

A `Num`/`Str`/`Leaf` tree below is the single source of truth for one tree's
semantics. Two pure converters turn it into:

  * a `decider2.trees.schema.Tree` document -> real `emit_tree()` source ->
    real numba compile (the codegen path decider2 ships today), and
  * flat struct-of-arrays numpy arrays for the ONE generic interpreted
    kernel in `interpreted_kernel.py`.

Building both from the same object is what makes the "identical answers"
assertion in run_experiment.py meaningful rather than accidental.

Simplification made deliberately: each *string* feature is tested by at
most one node per tree. decider2's codegen hoists all uses of one string
feature into a single shared matcher step (`EmitContext.string_test`), and
reconstructing that shared step's exact literal order from outside
`codegen.py` (to drive the row-loop wrapper) is extra bookkeeping that
doesn't change what's being measured (kernel cost), so it's avoided by
construction instead. The credit tree (the only shape with strings) respects
this.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Canonical, engine-agnostic tree
# ---------------------------------------------------------------------------

OPS = ("<", "<=", "==", ">", ">=", "!=")
OP_CODE = {op: i for i, op in enumerate(OPS)}


@dataclass
class Leaf:
    value: float


@dataclass
class Num:
    feature: str
    op: str  # one of OPS
    threshold: float
    then: "Node"
    otherwise: "Node"


@dataclass
class Str:
    feature: str
    patterns: list  # list[str], OR'd
    then: "Node"
    otherwise: "Node"


Node = Union[Leaf, Num, Str]


@dataclass
class TreeShape:
    """A built tree plus the metadata needed to feed it realistic rows."""

    name: str
    root: Node
    numeric_ranges: dict  # feature -> (lo, hi), for row generation
    string_categories: dict  # feature -> list[str], code = list.index(value)
    leaf_count: int
    node_count: int  # internal + leaf


# ---------------------------------------------------------------------------
# Shape builders
# ---------------------------------------------------------------------------


def _counter():
    n = [0]

    def next_id():
        n[0] += 1
        return n[0] - 1

    return next_id


def build_full_binary(depth: int, *, n_features: int = 8, seed: int = 0) -> TreeShape:
    """Every internal node is a numeric threshold test; both arms recurse
    until `depth`. 2**depth leaves, 2**depth - 1 internal nodes."""
    rng = random.Random(seed)
    features = [f"f{i}" for i in range(n_features)]
    lo, hi = 0.0, 100.0
    leaf_id = _counter()
    node_count = [0]

    def build(d: int) -> Node:
        node_count[0] += 1
        if d == 0:
            return Leaf(float(leaf_id()))
        feature = features[rng.randrange(n_features)]
        threshold = rng.uniform(lo, hi)
        return Num(
            feature=feature, op="<", threshold=threshold,
            then=build(d - 1), otherwise=build(d - 1),
        )

    root = build(depth)
    return TreeShape(
        name=f"full_binary_d{depth}",
        root=root,
        numeric_ranges={f: (lo, hi) for f in features},
        string_categories={},
        leaf_count=2 ** depth,
        node_count=node_count[0],
    )


def build_one_sided_chain(n: int, *, n_features: int = 10, seed: int = 0) -> TreeShape:
    """An ordinary policy waterfall: rule_i either fires (a leaf, `then`)
    or falls through to rule_{i+1} (`otherwise`). Flat in emitted source
    (decider2's `child_lines` flattening), deep in call structure."""
    rng = random.Random(seed)
    features = [f"g{i}" for i in range(n_features)]
    lo, hi = 0.0, 100.0
    leaf_id = _counter()

    default_leaf = Leaf(float(leaf_id()))
    node = default_leaf
    nodes_built = 1
    for i in reversed(range(n)):
        feature = features[i % n_features]
        threshold = rng.uniform(lo, hi)
        op = rng.choice(["<", ">="])
        node = Num(
            feature=feature, op=op, threshold=threshold,
            then=Leaf(float(leaf_id())), otherwise=node,
        )
        nodes_built += 2
    return TreeShape(
        name=f"one_sided_chain_{n}",
        root=node,
        numeric_ranges={f: (lo, hi) for f in features},
        string_categories={},
        leaf_count=n + 1,
        node_count=nodes_built,
    )


# A realistic credit underwriting tree: mixed numeric thresholds and two
# categorical (string) tests, each on its own feature, each used once.
CREDIT_NUMERIC_RANGES = {
    "credit_score": (300.0, 850.0),
    "dti_ratio": (0.0, 1.0),
    "monthly_income": (0.0, 50000.0),
    "months_employed": (0.0, 480.0),
    "existing_defaults": (0.0, 5.0),
    "loan_to_value": (0.0, 1.5),
    "age": (18.0, 90.0),
    "months_on_book": (0.0, 240.0),
}
CREDIT_STRING_CATEGORIES = {
    "employment_status": ["employed", "self_employed", "unemployed", "retired"],
    "region": ["north", "south", "east", "west", "central"],
}


def build_credit_tree(*, target_leaves: int = 20, seed: int = 0) -> TreeShape:
    """~20-leaf tree mixing numeric thresholds and two string equality
    tests — the shape doc 05 calls out as the realistic one.

    Built by recursive leaf-budget splitting: a node with `budget` leaves
    below it splits into two children whose budgets sum to `budget`, until
    budget == 1 (a leaf). Structure is fixed by the split; only which
    feature/op/threshold a node tests is randomised (seeded).
    """
    rng = random.Random(seed)
    numeric_features = list(CREDIT_NUMERIC_RANGES)
    string_features = list(CREDIT_STRING_CATEGORIES)
    leaf_id = _counter()
    node_count = [0]
    used_string_features: set = set()

    def build(budget: int) -> Node:
        node_count[0] += 1
        if budget <= 1:
            return Leaf(float(leaf_id()))
        left_budget = budget // 2
        right_budget = budget - left_budget
        # ~20% of internal nodes test a (still-unused) string feature.
        use_string = (
            rng.random() < 0.2
            and len(used_string_features) < len(string_features)
        )
        if use_string:
            remaining = [f for f in string_features if f not in used_string_features]
            feature = rng.choice(remaining)
            used_string_features.add(feature)
            cats = CREDIT_STRING_CATEGORIES[feature]
            k = rng.randint(1, max(1, len(cats) - 1))
            patterns = rng.sample(cats, k)
            return Str(
                feature=feature, patterns=patterns,
                then=build(left_budget), otherwise=build(right_budget),
            )
        feature = rng.choice(numeric_features)
        lo, hi = CREDIT_NUMERIC_RANGES[feature]
        threshold = rng.uniform(lo, hi)
        op = rng.choice(["<", "<=", ">", ">=", "=="])
        # '==' only makes sense for the near-integer-count features.
        if op == "==" and feature not in ("existing_defaults",):
            op = rng.choice(["<", ">="])
        if op == "==":
            threshold = float(rng.randint(0, 3))
        return Num(
            feature=feature, op=op, threshold=threshold,
            then=build(left_budget), otherwise=build(right_budget),
        )

    root = build(target_leaves)
    return TreeShape(
        name="credit_tree",
        root=root,
        numeric_ranges=dict(CREDIT_NUMERIC_RANGES),
        string_categories={f: CREDIT_STRING_CATEGORIES[f] for f in used_string_features},
        leaf_count=target_leaves,
        node_count=node_count[0],
    )


# ---------------------------------------------------------------------------
# Converter 1: canonical tree -> decider2.trees.schema.Tree document
# ---------------------------------------------------------------------------


def to_schema_tree(shape: TreeShape):
    from decider2.trees import (
        LeafNode, MultiEdgeData, MultiSourceEdge, PositionedNode, Tree, TreeOutput,
        UnaryEqual, UnaryGreaterThan, UnaryGreaterThanEqual, UnaryLessThan,
        UnaryLessThanEqual, UnaryNotEqual, UnaryNode, UnaryStringMatch,
    )

    unary_cls = {
        "<": UnaryLessThan, "<=": UnaryLessThanEqual, "==": UnaryEqual,
        ">": UnaryGreaterThan, ">=": UnaryGreaterThanEqual, "!=": UnaryNotEqual,
    }

    nodes: list = []
    edges: list = []
    leaf_rows: list = []
    ids = _counter()

    def visit(node: Node) -> str:
        my_id = f"n{ids()}"
        if isinstance(node, Leaf):
            result_idx = len(leaf_rows)
            leaf_rows.append({"score": node.value})
            nodes.append(PositionedNode(id=my_id, data=LeafNode(result_idx=result_idx)))
            return my_id
        if isinstance(node, Num):
            cls = unary_cls[node.op]
            nodes.append(PositionedNode(
                id=my_id,
                data=UnaryNode(condition=cls(feature=node.feature, threshold=node.threshold)),
            ))
        elif isinstance(node, Str):
            nodes.append(PositionedNode(
                id=my_id,
                data=UnaryNode(condition=UnaryStringMatch(
                    feature=node.feature, patterns=list(node.patterns),
                )),
            ))
        else:
            raise TypeError(node)
        then_id = visit(node.then)
        otherwise_id = visit(node.otherwise)
        edges.append(MultiSourceEdge(source=my_id, target=then_id, data=MultiEdgeData(sourceIndex=[0])))
        edges.append(MultiSourceEdge(source=my_id, target=otherwise_id, data=MultiEdgeData(sourceIndex=[1])))
        return my_id

    visit(shape.root)
    return Tree(
        name=shape.name,
        nodes=nodes,
        edges=edges,
        output=TreeOutput(data=leaf_rows, dtypes=[("score", "Float64")], default={"score": 0.0}),
    )


# ---------------------------------------------------------------------------
# Converter 2: canonical tree -> flat struct-of-arrays for the interpreted
# kernel. Node-kind codes match interpreted_kernel.py: 0=leaf, 1=numeric,
# 2=string-membership.
# ---------------------------------------------------------------------------

KIND_LEAF, KIND_NUM, KIND_STR = 0, 1, 2


@dataclass
class FlatTree:
    kind: np.ndarray
    feat_idx: np.ndarray
    op_code: np.ndarray
    thresh: np.ndarray
    pat_start: np.ndarray
    pat_count: np.ndarray
    patterns: np.ndarray
    left: np.ndarray
    right: np.ndarray
    leaf_value: np.ndarray
    numeric_features: tuple  # column order for the numeric matrix
    string_features: tuple  # column order for the string-code matrix


def to_flat_tree(shape: TreeShape) -> FlatTree:
    numeric_features = sorted(shape.numeric_ranges)
    string_features = sorted(shape.string_categories)
    num_col = {f: i for i, f in enumerate(numeric_features)}
    str_col = {f: i for i, f in enumerate(string_features)}

    kind: list = []
    feat_idx: list = []
    op_code: list = []
    thresh: list = []
    pat_start: list = []
    pat_count: list = []
    patterns: list = []
    left: list = []
    right: list = []
    leaf_value: list = []

    def add_leaf(value: float) -> int:
        idx = len(kind)
        kind.append(KIND_LEAF)
        feat_idx.append(-1)
        op_code.append(-1)
        thresh.append(0.0)
        pat_start.append(-1)
        pat_count.append(0)
        left.append(-1)
        right.append(-1)
        leaf_value.append(value)
        return idx

    def add(node: Node) -> int:
        if isinstance(node, Leaf):
            return add_leaf(node.value)
        idx = len(kind)
        # reserve the slot before recursing, exactly like the jittree POC's
        # flatten_tree — children get real indices, this one gets patched.
        kind.append(-1)
        feat_idx.append(-1)
        op_code.append(-1)
        thresh.append(0.0)
        pat_start.append(-1)
        pat_count.append(0)
        left.append(-1)
        right.append(-1)
        leaf_value.append(0.0)

        if isinstance(node, Num):
            kind[idx] = KIND_NUM
            feat_idx[idx] = num_col[node.feature]
            op_code[idx] = OP_CODE[node.op]
            thresh[idx] = node.threshold
        elif isinstance(node, Str):
            kind[idx] = KIND_STR
            feat_idx[idx] = str_col[node.feature]
            cats = shape.string_categories[node.feature]
            start = len(patterns)
            for p in node.patterns:
                patterns.append(cats.index(p))
            pat_start[idx] = start
            pat_count[idx] = len(node.patterns)
        else:
            raise TypeError(node)

        left[idx] = add(node.then)
        right[idx] = add(node.otherwise)
        return idx

    add(shape.root)

    return FlatTree(
        kind=np.array(kind, dtype=np.int8),
        feat_idx=np.array(feat_idx, dtype=np.int32),
        op_code=np.array(op_code, dtype=np.int8),
        thresh=np.array(thresh, dtype=np.float64),
        pat_start=np.array(pat_start, dtype=np.int32),
        pat_count=np.array(pat_count, dtype=np.int32),
        patterns=np.array(patterns if patterns else [0], dtype=np.int32),
        left=np.array(left, dtype=np.int32),
        right=np.array(right, dtype=np.int32),
        leaf_value=np.array(leaf_value, dtype=np.float64),
        numeric_features=tuple(numeric_features),
        string_features=tuple(string_features),
    )


# ---------------------------------------------------------------------------
# Row generation — same rows feed both engines, by feature name.
# ---------------------------------------------------------------------------


def collect_string_patterns(shape: TreeShape) -> dict:
    """`{feature: [pattern, ...]}`, one entry per string feature (the
    single-use-per-feature simplification means this is exactly the one
    `Str` node's own `.patterns`, in the order codegen's hoisted matcher
    will register them — see the module docstring)."""
    out: dict = {}

    def visit(node: Node) -> None:
        if isinstance(node, Leaf):
            return
        if isinstance(node, Str):
            out[node.feature] = list(node.patterns)
        visit(node.then)
        visit(node.otherwise)

    visit(shape.root)
    return out


def make_rows(shape: TreeShape, n_rows: int, *, seed: int = 0):
    """`(numeric: dict[str, float64 array], string_codes: dict[str, int32 array])`."""
    rng = np.random.default_rng(seed)
    numeric = {
        f: rng.uniform(lo, hi, size=n_rows).astype(np.float64)
        for f, (lo, hi) in sorted(shape.numeric_ranges.items())
    }
    string_codes = {
        f: rng.integers(0, len(cats), size=n_rows).astype(np.int32)
        for f, cats in sorted(shape.string_categories.items())
    }
    return numeric, string_codes
