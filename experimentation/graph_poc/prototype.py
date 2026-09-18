"""POC: a numba-first computation graph where plain functions and decision
trees are both just `Node` implementations, wired together by matching
parameter names to other nodes' names (same convention as fun2(fun1, c)).

The design question this answers: some config changes the GRAPH'S STRUCTURE
(which nodes exist, a tree's shape/feature/op) and must trigger a recompile;
other config is just NUMBERS (thresholds, leaf values) that should update
with zero recompile. The split is implemented via numba's own dispatch-by-
type caching, not by hand-rolled bookkeeping:

- `slots`: one float64[:] array holding every leaf input and every node's
  output, at a fixed index assigned when the graph is compiled (structural).
- `params`: one float64[:] array holding every tunable numeric value across
  every node (thresholds, leaf outputs, ...), at a fixed offset assigned the
  same way (structural: WHICH offset), but the *value* stored there is free
  to change at every call (parametric).
- Every node compiles down to a closure of the exact same signature
  `(slots, params) -> float64`, so they all live in one `numba.typed.List`
  of first-class functions. The single generic driver `_run_graph` that
  loops over that list is written ONCE, here, and reused verbatim by every
  graph you build - numba's own type-based specialization cache is what
  gives you "compile once per distinct graph shape, free updates to values".

A `pydantic` model is generated per-graph purely for the human-facing side:
named, validated parameter fields. It has nothing to do with numba's cache;
it is just a friendlier way to build the flat `params` array.
"""
from __future__ import annotations

import inspect
import time
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numba import njit, types
from numba.typed import List as TypedList
from pydantic import BaseModel, create_model

_NODE_SIG = types.float64(types.float64[:], types.float64[:])


@njit(cache=True)
def _run_graph(slots, params, node_fns, out_slots):
    for k in range(len(node_fns)):
        slots[out_slots[k]] = node_fns[k](slots, params)


# ---------------------------------------------------------------------------
# Node protocol + two implementations
# ---------------------------------------------------------------------------


class Node(ABC):
    name: str

    @abstractmethod
    def input_names(self) -> t.List[str]:
        """Names this node needs - either another node's name (wired) or a
        leaf input supplied by the caller."""

    def param_spec(self) -> t.Dict[str, float]:
        """Named tunable float fields this node contributes, with defaults."""
        return {}

    @abstractmethod
    def build_closure(self, slot_of: t.Dict[str, int], param_offset: int):
        """Return an @njit(_NODE_SIG) closure: (slots, params) -> float64."""


def _wrap_arity(raw, idxs: t.Tuple[int, ...]):
    """Small, finite, hand-written arity table - avoids ever generating
    source text for a variable number of positional arguments."""
    n = len(idxs)
    if n == 0:
        @njit(_NODE_SIG)
        def _closure(slots, params):
            return raw()
    elif n == 1:
        i0 = idxs[0]

        @njit(_NODE_SIG)
        def _closure(slots, params):
            return raw(slots[i0])
    elif n == 2:
        i0, i1 = idxs

        @njit(_NODE_SIG)
        def _closure(slots, params):
            return raw(slots[i0], slots[i1])
    elif n == 3:
        i0, i1, i2 = idxs

        @njit(_NODE_SIG)
        def _closure(slots, params):
            return raw(slots[i0], slots[i1], slots[i2])
    else:
        raise NotImplementedError(
            f"FunctionNode supports up to 3 inputs in this POC ({n} given) - "
            "add another branch to _wrap_arity for more."
        )
    return _closure


class FunctionNode(Node):
    """Wraps a plain annotated scalar function, e.g. `def fun1(a, b): ...`.
    Purely structural - no tunable params of its own in this POC."""

    def __init__(self, fn: t.Callable[..., float]):
        self.fn = fn
        self.name = fn.__name__
        self._params = list(inspect.signature(fn).parameters)

    def input_names(self) -> t.List[str]:
        return list(self._params)

    def build_closure(self, slot_of: t.Dict[str, int], param_offset: int):
        raw = njit(self.fn)
        idxs = tuple(slot_of[p] for p in self._params)
        return _wrap_arity(raw, idxs)


@dataclass
class Split:
    feature: str
    op: t.Literal[">", "<", "=="]
    threshold: float
    on_true: "TreeSpec"
    on_false: "TreeSpec"


@dataclass
class Leaf:
    value: float


TreeSpec = t.Union[Split, Leaf]
_OP_CODE = {">": 0, "<": 1, "==": 2}


class TreeNode(Node):
    """A decision tree. Shape (topology/feature/op) is structural - baked
    into the closure as constants. Thresholds and leaf values are NOT baked
    in; they live in `params`, so retuning them never recompiles anything."""

    def __init__(self, name: str, spec: TreeSpec):
        self.name = name
        self._is_leaf: t.List[bool] = []
        self._feature: t.List[t.Optional[str]] = []
        self._op_code: t.List[int] = []
        self._left: t.List[int] = []
        self._right: t.List[int] = []
        self._defaults: t.List[float] = []
        self._add(spec)
        self._feature_names = sorted({f for f in self._feature if f is not None})

    def _add(self, node: TreeSpec) -> int:
        idx = len(self._is_leaf)
        self._is_leaf.append(isinstance(node, Leaf))
        self._feature.append(None)
        self._op_code.append(-1)
        self._left.append(-1)
        self._right.append(-1)
        self._defaults.append(0.0)
        if isinstance(node, Leaf):
            self._defaults[idx] = node.value
        else:
            self._feature[idx] = node.feature
            self._op_code[idx] = _OP_CODE[node.op]
            self._defaults[idx] = node.threshold
            self._left[idx] = self._add(node.on_true)
            self._right[idx] = self._add(node.on_false)
        return idx

    def input_names(self) -> t.List[str]:
        return self._feature_names

    def param_spec(self) -> t.Dict[str, float]:
        return {f"{self.name}_p{i}": v for i, v in enumerate(self._defaults)}

    def build_closure(self, slot_of: t.Dict[str, int], param_offset: int):
        is_leaf = np.array(self._is_leaf, dtype=np.bool_)
        feature_slot = np.array(
            [slot_of[f] if f is not None else -1 for f in self._feature], dtype=np.int64
        )
        op_code = np.array(self._op_code, dtype=np.int64)
        left = np.array(self._left, dtype=np.int64)
        right = np.array(self._right, dtype=np.int64)

        @njit(_NODE_SIG)
        def _closure(slots, params):
            node = 0
            while not is_leaf[node]:
                f = slots[feature_slot[node]]
                thresh = params[param_offset + node]
                op = op_code[node]
                if op == 0:
                    cond = f > thresh
                elif op == 1:
                    cond = f < thresh
                else:
                    cond = f == thresh
                node = left[node] if cond else right[node]
            return params[param_offset + node]

        return _closure


# ---------------------------------------------------------------------------
# Graph: wires Nodes by name, assigns slots/param offsets, compiles once.
# ---------------------------------------------------------------------------


class Graph:
    def __init__(self, *nodes: Node):
        self.nodes = list(nodes)
        self._by_name = {n.name: n for n in self.nodes}
        if len(self._by_name) != len(self.nodes):
            raise ValueError("duplicate node names in graph")

    def _topo_order(self) -> t.List[Node]:
        order: t.List[Node] = []
        visited: t.Set[str] = set()
        visiting: t.Set[str] = set()

        def visit(n: Node) -> None:
            if n.name in visited:
                return
            if n.name in visiting:
                raise ValueError(f"cycle detected at node {n.name!r}")
            visiting.add(n.name)
            for dep_name in n.input_names():
                dep = self._by_name.get(dep_name)
                if dep is not None:
                    visit(dep)
            visiting.discard(n.name)
            visited.add(n.name)
            order.append(n)

        for n in self.nodes:
            visit(n)
        return order

    def compile(self, output: str) -> "CompiledGraph":
        order = self._topo_order()

        slot_of: t.Dict[str, int] = {}
        leaf_names: t.List[str] = []
        for n in order:
            for dep_name in n.input_names():
                if dep_name not in self._by_name and dep_name not in slot_of:
                    slot_of[dep_name] = len(slot_of)
                    leaf_names.append(dep_name)
        for n in order:
            slot_of[n.name] = len(slot_of)

        param_defaults: t.Dict[str, float] = {}
        param_offset_of: t.Dict[str, int] = {}
        for n in order:
            param_offset_of[n.name] = len(param_defaults)
            param_defaults.update(n.param_spec())
        param_names_ordered = list(param_defaults.keys())

        params_model_name = "GraphParams_" + "_".join(n.name for n in order)
        ParamsModel = create_model(
            params_model_name,
            **{k: (float, v) for k, v in param_defaults.items()},
        )

        node_fns = TypedList.empty_list(types.FunctionType(_NODE_SIG))
        out_slots: t.List[int] = []
        for n in order:
            node_fns.append(n.build_closure(slot_of, param_offset_of[n.name]))
            out_slots.append(slot_of[n.name])

        if output not in slot_of:
            raise ValueError(f"unknown output {output!r}")

        return CompiledGraph(
            leaf_names=leaf_names,
            slot_of=dict(slot_of),
            node_fns=node_fns,
            out_slots=np.array(out_slots, dtype=np.int64),
            output_slot=slot_of[output],
            n_slots=len(slot_of),
            ParamsModel=ParamsModel,
            param_names_ordered=param_names_ordered,
        )


@dataclass
class CompiledGraph:
    leaf_names: t.List[str]
    slot_of: t.Dict[str, int]
    node_fns: t.Any
    out_slots: np.ndarray
    output_slot: int
    n_slots: int
    ParamsModel: t.Type[BaseModel]
    param_names_ordered: t.List[str]

    def pack_params(self, params: t.Union[None, BaseModel, dict]) -> np.ndarray:
        if params is None:
            params = {}
        if not isinstance(params, self.ParamsModel):
            params = self.ParamsModel(**params)
        return np.array(
            [getattr(params, name) for name in self.param_names_ordered], dtype=np.float64
        )

    def __call__(self, params: t.Union[None, BaseModel, dict] = None, **leaf_values: float) -> float:
        params_arr = self.pack_params(params)
        slots = np.zeros(self.n_slots, dtype=np.float64)
        for name in self.leaf_names:
            slots[self.slot_of[name]] = leaf_values[name]
        _run_graph(slots, params_arr, self.node_fns, self.out_slots)
        return float(slots[self.output_slot])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    def fun1(a: float, b: float) -> float:
        return a + b

    def fun2(fun1: float, c: float) -> float:
        return fun1 * c

    risk_tier = TreeNode(
        "risk_tier",
        Split(
            feature="income", op=">", threshold=5000.0,
            on_true=Leaf(1.0),
            on_false=Split(
                feature="income", op="<", threshold=1000.0,
                on_true=Leaf(-1.0),
                on_false=Leaf(0.0),
            ),
        ),
    )

    def combine(fun2: float, risk_tier: float) -> float:
        return fun2 + risk_tier

    graph = Graph(FunctionNode(fun1), FunctionNode(fun2), risk_tier, FunctionNode(combine))
    compiled = graph.compile(output="combine")

    print("auto-collected params schema:", list(compiled.ParamsModel.model_fields))

    t0 = time.perf_counter()
    result = compiled(a=2.0, b=3.0, c=4.0, income=6000.0)
    print(f"first call  (compiles): result={result!r}  {1e3 * (time.perf_counter() - t0):.2f}ms")

    t0 = time.perf_counter()
    result = compiled(a=2.0, b=3.0, c=4.0, income=6000.0)
    print(f"second call (cached):   result={result!r}  {1e6 * (time.perf_counter() - t0):.2f}us")
    print("  _run_graph specializations so far:", len(_run_graph.signatures))

    # Retune a threshold - a VALUE change, not a structural one.
    tuned = compiled.ParamsModel(risk_tier_p0=10_000.0)  # raise the ">5000" split to ">10000"
    t0 = time.perf_counter()
    result = compiled(a=2.0, b=3.0, c=4.0, income=6000.0, params=tuned)
    print(f"tuned params (no recompile): result={result!r}  {1e6 * (time.perf_counter() - t0):.2f}us")
    print("  _run_graph specializations after tuning:", len(_run_graph.signatures))


if __name__ == "__main__":
    main()
