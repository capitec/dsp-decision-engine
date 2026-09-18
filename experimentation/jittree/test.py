"""POC: build a decision tree from a config, then compile it four ways with numba
and compare "compile once, rerun fast" behaviour.

Approach A (codegen):       config -> Python source (nested if/else, constants
                             baked in) -> exec -> njit. One compile PER DISTINCT
                             CONFIG, then calls are a pure branch chain.
Approach B (flat arrays):   config -> flattened numpy arrays (sklearn-tree style)
                             -> ONE generic njit walker that loops over the
                             arrays. The walker compiles exactly once, ever; new
                             configs are just new arrays, no recompilation.
Approach C (dispatch table): like B, but the per-op comparisons (>, <, ==) are
                             themselves separately njit-compiled functions, put
                             into a numba typed.List of first-class function
                             types. The generic walker calls them by index. New
                             operators = one new tiny jitted function; the walker
                             never changes.
Approach D (closures):      no text/exec at all. The tree is compiled bottom-up
                             into nested njit closures - each node closes over
                             its threshold/feature index and calls its already-
                             compiled child closures directly. One compile PER
                             DISTINCT CONFIG (like A), but built purely through
                             Python function composition.
"""

import time
import typing as t

import numpy as np
from numba import njit, types
from numba.typed import List as TypedList
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config schema: parsing / validation only. Not used directly at eval time.
# ---------------------------------------------------------------------------


class GTNode(BaseModel):
    op: t.Literal[">"] = ">"
    feature: str
    thresh: float
    on_true: "NodeConfig"
    on_false: "NodeConfig"


class LTNode(BaseModel):
    op: t.Literal["<"] = "<"
    feature: str
    thresh: float
    on_true: "NodeConfig"
    on_false: "NodeConfig"


class EQNode(BaseModel):
    op: t.Literal["=="] = "=="
    feature: str
    thresh: float
    on_true: "NodeConfig"
    on_false: "NodeConfig"


class LeafNode(BaseModel):
    op: t.Literal["leaf"] = "leaf"
    output: float


NodeConfig = t.Annotated[
    t.Union[GTNode, LTNode, EQNode, LeafNode], Field(discriminator="op")
]

for _cls in (GTNode, LTNode, EQNode, LeafNode):
    _cls.model_rebuild()


def parse_tree(config: dict) -> "GTNode | LTNode | EQNode | LeafNode":
    from pydantic import TypeAdapter

    return TypeAdapter(NodeConfig).validate_python(config)


def collect_features(node) -> list[str]:
    """Depth-first, first-seen order -> stable feature -> index mapping."""
    seen: dict[str, None] = {}

    def walk(n) -> None:
        if isinstance(n, LeafNode):
            return
        seen.setdefault(n.feature, None)
        walk(n.on_true)
        walk(n.on_false)

    walk(node)
    return list(seen)


# ---------------------------------------------------------------------------
# Approach A: codegen a straight-line function, then njit it.
# ---------------------------------------------------------------------------


def _emit(node, feature_index: dict[str, int], indent: int) -> list[str]:
    pad = "    " * indent
    if isinstance(node, LeafNode):
        return [f"{pad}return {node.output!r}"]
    idx = feature_index[node.feature]
    lines = [f"{pad}if x[{idx}] {node.op} {node.thresh!r}:"]
    lines += _emit(node.on_true, feature_index, indent + 1)
    lines.append(f"{pad}else:")
    lines += _emit(node.on_false, feature_index, indent + 1)
    return lines


def compile_tree_codegen(root, feature_index: dict[str, int]):
    """Returns (njit-compiled function, generated source) for this exact tree."""
    body = "\n".join(_emit(root, feature_index, 1))
    src = f"def _tree_fn(x):\n{body}\n"
    namespace: dict = {}
    exec(src, namespace)
    # exec'd code has no real source file, so on-disk caching (cache=True)
    # isn't available here - numba still keeps the compiled function in
    # memory for the life of the process, which is all this POC needs.
    fn = njit(namespace["_tree_fn"])
    return fn, src


# ---------------------------------------------------------------------------
# Approach B: flatten to arrays, one generic njit walker for any tree.
# ---------------------------------------------------------------------------

_OP_CODE = {">": 0, "<": 1, "==": 2}


def flatten_tree(root, feature_index: dict[str, int]):
    is_leaf: list[bool] = []
    feature_idx: list[int] = []
    op_code: list[int] = []
    thresh: list[float] = []
    left: list[int] = []
    right: list[int] = []
    leaf_value: list[float] = []

    def add(node) -> int:
        idx = len(is_leaf)
        is_leaf.append(isinstance(node, LeafNode))
        feature_idx.append(-1)
        op_code.append(-1)
        thresh.append(0.0)
        left.append(-1)
        right.append(-1)
        leaf_value.append(0.0)
        if isinstance(node, LeafNode):
            leaf_value[idx] = node.output
        else:
            feature_idx[idx] = feature_index[node.feature]
            op_code[idx] = _OP_CODE[node.op]
            thresh[idx] = node.thresh
            left[idx] = add(node.on_true)
            right[idx] = add(node.on_false)
        return idx

    add(root)
    return (
        np.array(is_leaf, dtype=np.bool_),
        np.array(feature_idx, dtype=np.int64),
        np.array(op_code, dtype=np.int64),
        np.array(thresh, dtype=np.float64),
        np.array(left, dtype=np.int64),
        np.array(right, dtype=np.int64),
        np.array(leaf_value, dtype=np.float64),
    )


@njit(cache=True)
def _walk(x, is_leaf, feature_idx, op_code, thresh, left, right, leaf_value):
    node = 0
    while not is_leaf[node]:
        f = x[feature_idx[node]]
        t_ = thresh[node]
        op = op_code[node]
        if op == 0:
            cond = f > t_
        elif op == 1:
            cond = f < t_
        else:
            cond = f == t_
        node = left[node] if cond else right[node]
    return leaf_value[node]


class CompiledTreeArrays:
    """Same compiled `_walk` reused across any number of different trees."""

    def __init__(self, root, feature_index: dict[str, int]):
        self.feature_index = feature_index
        self.arrays = flatten_tree(root, feature_index)

    def __call__(self, x: np.ndarray) -> float:
        return _walk(x, *self.arrays)


# ---------------------------------------------------------------------------
# Approach C: prejitted per-op functions in a numba dispatch table, called by
# index from a single generic njit walker (same array layout as Approach B).
# ---------------------------------------------------------------------------

_CMP_SIG = types.boolean(types.float64, types.float64)


@njit(_CMP_SIG)
def _op_gt(f, thresh):
    return f > thresh


@njit(_CMP_SIG)
def _op_lt(f, thresh):
    return f < thresh


@njit(_CMP_SIG)
def _op_eq(f, thresh):
    return f == thresh


# order must match _OP_CODE (">"=0, "<"=1, "=="=2)
_OPS = TypedList.empty_list(types.FunctionType(_CMP_SIG))
_OPS.append(_op_gt)
_OPS.append(_op_lt)
_OPS.append(_op_eq)


@njit(cache=True)
def _walk_dispatch(x, is_leaf, feature_idx, op_code, thresh, left, right, leaf_value, ops):
    node = 0
    while not is_leaf[node]:
        f = x[feature_idx[node]]
        cond = ops[op_code[node]](f, thresh[node])
        node = left[node] if cond else right[node]
    return leaf_value[node]


class CompiledTreeDispatch:
    """Like CompiledTreeArrays, but comparisons go through the `_OPS` table.

    Adding a new operator later means adding one more tiny @njit function to
    _OPS - `_walk_dispatch` itself never has to change or recompile.
    """

    def __init__(self, root, feature_index: dict[str, int]):
        self.feature_index = feature_index
        self.arrays = flatten_tree(root, feature_index)

    def __call__(self, x: np.ndarray) -> float:
        return _walk_dispatch(x, *self.arrays, _OPS)


# ---------------------------------------------------------------------------
# Approach D: no text/exec at all. Compile the tree bottom-up into nested
# njit closures - each node closes over its constants and calls its already
# -compiled children directly.
# ---------------------------------------------------------------------------


def compile_tree_closures(node, feature_index: dict[str, int]):
    if isinstance(node, LeafNode):
        value = node.output

        @njit
        def _leaf(x):
            return value

        return _leaf

    idx = feature_index[node.feature]
    thresh = node.thresh
    true_fn = compile_tree_closures(node.on_true, feature_index)
    false_fn = compile_tree_closures(node.on_false, feature_index)

    if node.op == ">":

        @njit
        def _node(x):
            if x[idx] > thresh:
                return true_fn(x)
            return false_fn(x)

    elif node.op == "<":

        @njit
        def _node(x):
            if x[idx] < thresh:
                return true_fn(x)
            return false_fn(x)

    else:

        @njit
        def _node(x):
            if x[idx] == thresh:
                return true_fn(x)
            return false_fn(x)

    return _node


# ---------------------------------------------------------------------------
# Demo / benchmark
# ---------------------------------------------------------------------------

EXAMPLE_CONFIG = {
    "op": ">",
    "feature": "a",
    "thresh": 5.0,
    "on_true": {
        "op": "<",
        "feature": "b",
        "thresh": 2.0,
        "on_true": {"op": "leaf", "output": 1.0},
        "on_false": {"op": "leaf", "output": 0.0},
    },
    "on_false": {
        "op": "==",
        "feature": "a",
        "thresh": 0.0,
        "on_true": {"op": "leaf", "output": -1.0},
        "on_false": {"op": "leaf", "output": -2.0},
    },
}


def main() -> None:
    root = parse_tree(EXAMPLE_CONFIG)
    features = collect_features(root)
    feature_index = {name: i for i, name in enumerate(features)}
    print("features:", feature_index)

    codegen_fn, src = compile_tree_codegen(root, feature_index)
    print("\n--- generated source (Approach A) ---")
    print(src)

    array_tree = CompiledTreeArrays(root, feature_index)
    dispatch_tree = CompiledTreeDispatch(root, feature_index)
    closure_fn = compile_tree_closures(root, feature_index)

    x = np.array([6.0, 1.0])

    approaches = (
        ("A codegen", codegen_fn),
        ("B flat-array", array_tree),
        ("C dispatch", dispatch_tree),
        ("D closures", closure_fn),
    )
    for label, fn in approaches:
        t0 = time.perf_counter()
        result = fn(x)
        compile_s = time.perf_counter() - t0

        n = 200_000
        t0 = time.perf_counter()
        for _ in range(n):
            fn(x)
        run_s = time.perf_counter() - t0

        print(
            f"{label:>10}: result={result!r}  "
            f"first_call={compile_s * 1e3:.3f}ms  "
            f"{n} reruns={run_s * 1e3:.3f}ms "
            f"({run_s / n * 1e9:.1f}ns/call)"
        )


if __name__ == "__main__":
    main()
