"""A stand-in for the decider authoring API and debug session.

Just enough of IR.md (step, dag, flow, branch, param, missing_as, to_ir, walk)
and Design.md's Session to drive the VS Code extension until engine/debug
lands. Batches are lists; a branch runs each arm on the rows that chose it.
"""
from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


class ParamSpec:
    __slots__ = ("default",)

    def __init__(self, default):
        self.default = default


class MissingAs:
    __slots__ = ("fill",)

    def __init__(self, fill):
        self.fill = fill


def param(default=None, **_constraints):
    return ParamSpec(default)


def missing_as(fill):
    return MissingAs(fill)


def _callsite() -> tuple[str, int]:
    f = sys._getframe(2)
    return f.f_code.co_filename, f.f_lineno


# ---------------------------------------------------------------- IR

@dataclass(frozen=True, slots=True)
class Origin:
    path: str
    source: str
    file: str | None = None
    line: int | None = None


@dataclass(eq=False)
class CallNode:
    origin: Origin
    fn: Callable
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    params: dict[str, Any]
    fills: dict[str, Any]

    def children(self):
        return ()


@dataclass(eq=False)
class SequenceNode:
    origin: Origin
    children_: tuple

    def children(self):
        return self.children_


@dataclass(eq=False)
class BranchNode:
    origin: Origin
    condition: CallNode
    arms: tuple
    modifies: tuple[str, ...]

    def children(self):
        return (self.condition, *self.arms)


@dataclass(frozen=True, slots=True)
class IRContext:
    path: str = ""

    def child(self, name):
        if name is None:
            return self
        return IRContext(f"{self.path}/{name}" if self.path else name)


# ---------------------------------------------------------------- steps

class Step:
    name: str | None

    def __or__(self, other):
        other = as_step(other)
        left = self.members if isinstance(self, SequentialStep) and self.name is None else (self,)
        return SequentialStep(None, (*left, other), *_callsite())

    def __ror__(self, other):
        return as_step(other) | self

    def walk(self, prefix="") -> Iterator[tuple[str, "Step"]]:
        path = f"{prefix}/{self.name}" if prefix and self.name else (self.name or prefix)
        yield path, self
        for m in getattr(self, "members", ()):
            yield from m.walk(path)

    def to_ir(self, ctx: IRContext):
        raise NotImplementedError

    def session(self, data, params=None):
        return Session(to_ir(self), data, params)

    def run(self, data, params=None):
        s = self.session(data, params)
        s.resume()
        return s.state.columns


def harvest(fn):
    inputs, params, fills = [], {}, {}
    for name, p in inspect.signature(fn).parameters.items():
        if isinstance(p.default, ParamSpec):
            params[name] = p.default.default
        elif isinstance(p.default, MissingAs):
            inputs.append(name)
            fills[name] = p.default.fill
        else:
            inputs.append(name)
    return tuple(inputs), params, fills


@dataclass(eq=False)
class FunctionStep(Step):
    name: str
    fn: Callable
    outputs: tuple[str, ...]

    def to_ir(self, ctx):
        inputs, params, fills = harvest(self.fn)
        code = self.fn.__code__
        origin = Origin(ctx.child(self.name).path, f"{self.fn.__module__}:{self.fn.__qualname__}",
                        code.co_filename, code.co_firstlineno)
        return CallNode(origin, self.fn, inputs, self.outputs, params, fills)


def step(fn=None, /, *, name=None, output=None, outputs=None):
    def wrap(f):
        outs = outputs or ((output,) if output else (f.__name__,))
        return FunctionStep(name or f.__name__, f, tuple(outs))
    return wrap(fn) if fn is not None else wrap


def as_step(x) -> Step:
    return x if isinstance(x, Step) else step(x)


@dataclass(eq=False)
class SequentialStep(Step):
    name: str | None
    members: tuple
    file: str
    line: int

    def to_ir(self, ctx):
        c = ctx.child(self.name)
        return SequenceNode(Origin(c.path, "decider.steps:SequentialStep", self.file, self.line),
                            tuple(m.to_ir(c) for m in self.members))


@dataclass(eq=False)
class DagStep(Step):
    name: str | None
    members: tuple
    file: str
    line: int

    def to_ir(self, ctx):
        c = ctx.child(self.name)
        nodes = [m.to_ir(c) for m in self.members]
        return SequenceNode(Origin(c.path, "decider.steps:DagStep", self.file, self.line), _toposort(nodes))


def _toposort(nodes):
    # ponytail: only CallNode members; nested combinators inside a dag keep written order
    written = {o: n for n in nodes if isinstance(n, CallNode) for o in n.outputs}
    done, order = set(), []

    def visit(n):
        if id(n) in done:
            return
        done.add(id(n))
        if isinstance(n, CallNode):
            for i in n.inputs:
                if i in written and written[i] is not n:
                    visit(written[i])
        order.append(n)

    for n in nodes:
        visit(n)
    return tuple(order)


@dataclass(eq=False)
class BranchStep(Step):
    name: str
    condition: Step
    arms: tuple
    modifies: tuple[str, ...]
    file: str
    line: int

    @property
    def members(self):
        return (self.condition, *self.arms)

    def to_ir(self, ctx):
        c = ctx.child(self.name)
        return BranchNode(Origin(c.path, "decider.steps:BranchStep", self.file, self.line),
                          self.condition.to_ir(c), tuple(a.to_ir(c) for a in self.arms), self.modifies)


def dag(*steps, name=None):
    return DagStep(name, tuple(as_step(s) for s in steps), *_callsite())


def flow(*steps, name=None):
    return SequentialStep(name, tuple(as_step(s) for s in steps), *_callsite())


def branch(condition, *arms, modifies, name):
    return BranchStep(name, as_step(condition), tuple(as_step(a) for a in arms), tuple(modifies), *_callsite())


def to_ir(step_: Step):
    return step_.to_ir(IRContext())


def call_nodes(node) -> list[CallNode]:
    """Every CallNode under `node`, in execution order (both branch arms)."""
    if isinstance(node, CallNode):
        return [node]
    return [c for ch in node.children() for c in call_nodes(ch)]


# ---------------------------------------------------------------- runtime

class State:
    def __init__(self, data):
        rows = data if isinstance(data, list) else [dict(zip(data, vals)) for vals in zip(*data.values())]
        self.n = len(rows)
        self.columns: dict[str, list] = {}
        self.versions: dict[str, list[tuple[str, list]]] = {}
        for k in sorted({k for r in rows for k in r}):
            self.write(k, [r.get(k) for r in rows], "input")

    def write(self, name, values, producer):
        self.columns[name] = list(values)
        self.versions.setdefault(name, []).append((producer, list(values)))

    def summary(self, name):
        col = self.columns[name]
        kinds = {type(v).__name__ for v in col if v is not None}
        return {"name": name, "dtype": "/".join(sorted(kinds)) or "null", "rows": self.n,
                "nulls": sum(v is None for v in col), "preview": col[:5],
                "producer": self.versions[name][-1][0], "versions": len(self.versions[name])}


@dataclass(frozen=True, slots=True)
class Checkpoint:
    path: str
    phase: str  # "start" | "end"
    depth: int
    node: Any = field(repr=False)


def _run_call(node: CallNode, state: State, params: dict, mask):
    outs = {o: list(state.columns.get(o, [None] * state.n)) for o in node.outputs}
    for row in range(state.n):
        if mask is not None and not mask[row]:
            continue
        args = {}
        for name in node.inputs:
            if name not in state.columns:
                raise KeyError(f"{node.origin.path}: input '{name}' is not in the data or produced earlier")
            v = state.columns[name][row]
            args[name] = node.fills.get(name) if v is None and name in node.fills else v
        result = node.fn(**args, **params)
        if len(node.outputs) == 1:
            result = (result,)
        for o, v in zip(node.outputs, result):
            outs[o][row] = v
    for o, vals in outs.items():
        state.write(o, vals, node.origin.path)


def iterate(node, state: State, params_doc: dict, ctl: dict, depth=0, mask=None) -> Iterator[Checkpoint]:
    path = node.origin.path
    yield Checkpoint(path, "start", depth, node)
    if isinstance(node, CallNode):
        if ctl.get("skip") == path:
            ctl["skip"] = None
        if ctl.get("skip") is None:
            _run_call(node, state, {**node.params, **params_doc.get(path, {})}, mask)
    elif isinstance(node, SequenceNode):
        for child in node.children_:
            yield from iterate(child, state, params_doc, ctl, depth + 1, mask)
    else:
        yield from iterate(node.condition, state, params_doc, ctl, depth + 1, mask)
        cond = state.columns[node.condition.outputs[0]]
        for i, arm in enumerate(node.arms):
            chosen = [(mask is None or mask[r]) and _arm_index(cond[r]) == i for r in range(state.n)]
            if any(chosen):
                yield from iterate(arm, state, params_doc, ctl, depth + 1, chosen)
    yield Checkpoint(path, "end", depth, node)


def _arm_index(value):
    if isinstance(value, bool):
        return 0 if value else 1
    return int(value)


class Session:
    """Drive a run one checkpoint at a time, as Design.md describes.

    Example::

        s = Session(to_ir(pipeline), rows)
        s.break_at("term/cap_by_income"); s.resume()
        s.set("term_cap", 12.0); s.resume()
    """

    def __init__(self, ir, data, params=None):
        self.ir = ir
        self.state = State(data)
        self.params = params or {}
        self.events: list[dict] = []
        self.breakpoints: set[str] = set()
        self.current: Checkpoint | None = None
        self.finished = False
        self._ctl: dict = {"skip": None}
        self._gen = iterate(ir, self.state, self.params, self._ctl)
        self._emit("RunStarted", rows=self.state.n)

    def _emit(self, event, **payload):
        self.events.append({"event": event, **payload})

    def break_at(self, path):
        self.breakpoints.add(path)

    def clear_break(self, path):
        self.breakpoints.discard(path)

    def _at_breakpoint(self):
        cp = self.current
        return cp is not None and cp.phase == "start" and any(
            cp.path == b or cp.path.startswith(b + "/") for b in self.breakpoints)

    def _advance(self) -> bool:
        try:
            cp = next(self._gen)
        except StopIteration:
            self.finished = True
            self._emit("RunFinished", columns=sorted(self.state.columns))
            return False
        self.current = cp
        if self._ctl["skip"] is not None:
            return True
        if cp.phase == "start":
            self._emit("NodeStarted", path=cp.path)
        elif isinstance(cp.node, CallNode):
            self._emit("NodeFinished", path=cp.path, outputs=[self.state.summary(o) for o in cp.node.outputs])
        else:
            self._emit("NodeFinished", path=cp.path, outputs=[])
        return True

    def step_into(self):
        while self._advance():
            if self.current.phase == "start" and self._ctl["skip"] is None:
                break
        self._paused("step")

    def step(self):
        depth = self.current.depth if self.current else 0
        while self._advance():
            if self.current.phase == "start" and self.current.depth <= depth:
                break
        self._paused("step")

    def resume(self):
        while self._advance():
            if self._at_breakpoint():
                break
        self._paused("breakpoint")

    def _paused(self, reason):
        if not self.finished:
            self._emit("Paused", path=self.current.path, reason=reason)

    def set(self, name, value):
        values = value if isinstance(value, list) else [value] * self.state.n
        producer = f"override@{self.current.path if self.current else ''}"
        self.state.write(name, values, producer)
        self._emit("Overridden", name=name, producer=producer)

    def rewind(self, path):
        self._ctl["skip"] = path
        self._gen = iterate(self.ir, self.state, self.params, self._ctl)
        self.finished = False
        while self._advance():
            if self.current.path == path and self.current.phase == "start":
                self._ctl["skip"] = None
                break
        self._emit("NodeStarted", path=path)
        self._paused("rewind")


def lineage(ir, name, before_path=None, depth=4, seen=None):
    """Static backward slice: which node last writes `name` before `before_path`, and what fed it."""
    seen = seen if seen is not None else set()
    nodes = call_nodes(ir)
    if before_path is not None:
        idx = next((i for i, n in enumerate(nodes) if n.origin.path == before_path), len(nodes))
        nodes = nodes[:idx]
    producer = next((n for n in reversed(nodes) if name in n.outputs), None)
    entry = {"name": name, "producer": producer.origin.path if producer else None, "inputs": []}
    if producer and depth > 0 and (name, producer.origin.path) not in seen:
        seen.add((name, producer.origin.path))
        entry["inputs"] = [lineage(ir, i, producer.origin.path, depth - 1, seen) for i in producer.inputs]
    return entry
