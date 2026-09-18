"""POC #2 - supersedes the slots/params design in prototype.py.

Design change: don't force every node into one common signature. Auto-wrap
each user function as-is (njit infers whatever types it actually needs -
float, int, array, numba-typed containers, ...) and only generate the thin
call-sequence glue (topological order + name-based wiring, same convention
as fun2(fun1, c)). That removes both the arity table and the "everything
must be one homogeneous float64 array" ceiling from the first prototype.

Two independent layers of graceful degradation, so you're never REQUIRED to
write numba-friendly code, you just don't get the speedup where you don't:

1. Per node (`NodeCallable`): try the jitted version; the first time numba
   can't compile it for the given argument types, permanently switch that
   one node to plain Python. Every other node is unaffected.
2. Per pipeline (`CompiledPipeline`): try to njit the WHOLE call-sequence
   glue, calling every node's raw njit dispatcher directly (maximum fusion -
   numba inlines njit->njit calls, so this is the fastest possible tier).
   If any single node isn't nopython-compatible, that fusion attempt fails
   as a whole; fall back to a plain-Python glue that calls each node through
   its own NodeCallable instead - so the nodes that DO jit still run fast,
   only the incompatible one(s) pay Python-call overhead.

Only `numba.core.errors.NumbaError` is treated as "fall back to slower" -
a genuine runtime bug (e.g. ZeroDivisionError) always propagates, jitted or
not, so it can't be silently masked as "this needed a fallback".
"""
import inspect
import re
import time
import typing as t

from numba import njit
from numba.core.errors import NumbaError


class NodeCallable:
    def __init__(self, fn: t.Callable):
        self.name = fn.__name__
        self._plain = fn
        self._jit = njit(fn)
        self.mode = "jit"

    def __call__(self, *args):
        if self.mode == "jit":
            try:
                return self._jit(*args)
            except NumbaError:
                self.mode = "plain"
        return self._plain(*args)


def _input_names(fn: t.Callable) -> t.List[str]:
    return list(inspect.signature(fn).parameters)


def _data_input_names(fn: t.Callable) -> t.List[str]:
    """Inputs that participate in wiring/leaf resolution - excludes the
    reserved `params` name, which the pipeline injects directly instead."""
    return [p for p in _input_names(fn) if p != "params"]


def _topo_order(fns: t.Sequence[t.Callable]) -> t.List[t.Callable]:
    by_name = {fn.__name__: fn for fn in fns}
    order: t.List[t.Callable] = []
    visited: t.Set[str] = set()
    visiting: t.Set[str] = set()

    def visit(fn: t.Callable) -> None:
        if fn.__name__ in visited:
            return
        if fn.__name__ in visiting:
            raise ValueError(f"cycle detected at {fn.__name__!r}")
        visiting.add(fn.__name__)
        for p in _data_input_names(fn):
            dep = by_name.get(p)
            if dep is not None:
                visit(dep)
        visiting.discard(fn.__name__)
        visited.add(fn.__name__)
        order.append(fn)

    for fn in fns:
        visit(fn)
    return order


class Pipeline:
    """Auto-wraps `*functions`, wiring them by matching parameter names to
    other functions' names."""

    def __init__(self, *functions: t.Callable):
        self.order = _topo_order(functions)
        self._nodes = {fn.__name__: NodeCallable(fn) for fn in self.order}

    def compile(self, output: str) -> "CompiledPipeline":
        by_name = {fn.__name__: fn for fn in self.order}
        uses_params = any("params" in _input_names(fn) for fn in self.order)

        leaf_names: t.List[str] = []
        seen: t.Set[str] = set()
        for fn in self.order:
            for p in _data_input_names(fn):
                if p not in by_name and p not in seen:
                    seen.add(p)
                    leaf_names.append(p)

        signature = leaf_names + (["params"] if uses_params else [])
        lines = [f"def _pipeline({', '.join(signature)}):"]
        for fn in self.order:
            call_args = [
                "params" if p == "params" else (f"v_{p}" if p in by_name else p)
                for p in _input_names(fn)
            ]
            lines.append(f"    v_{fn.__name__} = _{fn.__name__}({', '.join(call_args)})")
        lines.append(f"    return v_{output}")
        source = "\n".join(lines)

        fused_namespace = {f"_{name}": node._jit for name, node in self._nodes.items()}
        exec(source, fused_namespace)  # noqa: S102 - fixed shape, generated once per graph
        fused = njit(fused_namespace["_pipeline"])

        safe_namespace = {f"_{name}": node for name, node in self._nodes.items()}
        exec(source, safe_namespace)  # noqa: S102
        safe = safe_namespace["_pipeline"]

        return CompiledPipeline(
            fused=fused,
            safe=safe,
            leaf_names=leaf_names,
            uses_params=uses_params,
            source=source,
            nodes=self._nodes,
        )


class CompiledPipeline:
    def __init__(self, fused, safe, leaf_names, uses_params, source, nodes):
        self.fused = fused
        self.safe = safe
        self.leaf_names = leaf_names
        self.uses_params = uses_params
        self.source = source
        self.nodes = nodes
        self.mode = "fused"

    def __call__(self, params: t.Any = None, **leaf_values: t.Any):
        args = [leaf_values[n] for n in self.leaf_names]
        if self.uses_params:
            args.append(params)
        if self.mode == "fused":
            try:
                return self.fused(*args)
            except NumbaError:
                self.mode = "safe"
        return self.safe(*args)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    def fun1(a: float, b: float) -> float:
        return a + b

    def bad_node(code: str) -> float:
        # `re` isn't nopython-supported - numba can never compile this.
        m = re.match(r"[A-Z]+", code)
        return 1.0 if m else 0.0

    def combine(fun1: float, bad_node: float) -> float:
        return fun1 + bad_node

    print("=== mixed pipeline (one node numba can't touch) ===")
    mixed = Pipeline(fun1, bad_node, combine).compile(output="combine")
    print(mixed.source)

    t0 = time.perf_counter()
    result = mixed(a=2.0, b=3.0, code="ABC")
    print(f"first call:  result={result!r}  {1e3 * (time.perf_counter() - t0):.2f}ms  "
          f"pipeline.mode={mixed.mode!r}")
    print("  per-node modes:", {n: c.mode for n, c in mixed.nodes.items()})

    t0 = time.perf_counter()
    result = mixed(a=2.0, b=3.0, code="ABC")
    print(f"second call: result={result!r}  {1e6 * (time.perf_counter() - t0):.2f}us  "
          f"pipeline.mode={mixed.mode!r}")

    print("\n=== all-numba-friendly pipeline (should fully fuse) ===")

    def fun2(fun1: float, c: float) -> float:
        return fun1 * c

    clean = Pipeline(fun1, fun2).compile(output="fun2")
    t0 = time.perf_counter()
    result = clean(a=2.0, b=3.0, c=4.0)
    print(f"first call:  result={result!r}  {1e3 * (time.perf_counter() - t0):.2f}ms  "
          f"pipeline.mode={clean.mode!r}")
    t0 = time.perf_counter()
    result = clean(a=2.0, b=3.0, c=4.0)
    print(f"second call: result={result!r}  {1e6 * (time.perf_counter() - t0):.2f}us  "
          f"pipeline.mode={clean.mode!r}")

    print("\n=== a node with a `params` argument - retuned with zero recompiles ===")

    Params = t.NamedTuple("Params", [("threshold", float), ("bonus", float)])

    def scored(fun1: float, params) -> float:
        if fun1 > params.threshold:
            return fun1 + params.bonus
        return fun1

    tuned = Pipeline(fun1, scored).compile(output="scored")

    t0 = time.perf_counter()
    result = tuned(a=2.0, b=3.0, params=Params(threshold=4.0, bonus=10.0))
    print(f"first call  (compiles): result={result!r}  "
          f"{1e3 * (time.perf_counter() - t0):.2f}ms  mode={tuned.mode!r}")
    print("  fused.signatures:", len(tuned.fused.signatures))

    # Retune the threshold - a VALUE change, same Params type, no recompile.
    t0 = time.perf_counter()
    result = tuned(a=2.0, b=3.0, params=Params(threshold=10.0, bonus=10.0))
    print(f"retuned params (no recompile): result={result!r}  "
          f"{1e6 * (time.perf_counter() - t0):.2f}us  mode={tuned.mode!r}")
    print("  fused.signatures:", len(tuned.fused.signatures))


if __name__ == "__main__":
    main()
