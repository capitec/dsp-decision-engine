from __future__ import annotations

import ast
from types import ModuleType
from typing import Any, Iterator, Literal

from decider.checks._base import Check, CheckContext
from decider.checks._report import Finding

__all__ = ["Impure", "MutatesInput", "WallClock"]

_WALL_CLOCK: dict[str, set[str]] = {
    "time": {"time", "time_ns", "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns",
             "process_time", "process_time_ns", "thread_time", "thread_time_ns", "clock"},
    "datetime.datetime": {"now", "utcnow", "today"},
    "datetime.date": {"today"},
    "random": {"random", "randint", "randrange", "uniform", "choice", "choices", "shuffle",
               "sample", "gauss", "randbytes"},
    "uuid": {"uuid1", "uuid4"},
    "secrets": {"token_bytes", "token_hex", "token_urlsafe", "randbelow", "choice"},
    "os": {"urandom"},
}
_WALL_CLOCK_CALLS = {f"{owner}.{name}" for owner, names in _WALL_CLOCK.items() for name in names}

_MUTABLE_BUILTINS = {"list", "dict", "set"}
_MUTATING_METHODS = {"append", "extend", "insert", "pop", "remove", "clear", "update", "sort", "reverse", "add",
                     "discard", "setdefault", "popitem", "fill"}


class WallClock(Check):
    """Wall-clock and randomness reads (`time.time()`, `datetime.now()`, `random.random()`) make a flow non-deterministic."""

    type: Literal["wall_clock"] = "wall_clock"
    code = "DCD001"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node in ctx.calls():
            src = ctx.source(node)
            imports = _local_imports(src.tree) if src else {}
            for call in ast.walk(src.tree) if src else ():
                if isinstance(call, ast.Call) and _qualified(call.func, node.fn, imports) in _WALL_CLOCK_CALLS:
                    text = f"{ast.unparse(call.func)}()"
                    yield ctx.finding(node, f"{node.origin.path} reads {text}; a wall-clock read makes the flow "
                                      "non-deterministic", fix="pass the value in as an input column",
                                      line=src.line(call), call=text)


def _local_imports(tree: ast.AST) -> dict[str, str]:
    found: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                found[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                found[a.asname or a.name] = f"{n.module}.{a.name}"
    return found


def _qualified(func: ast.expr, fn: Any, imports: dict[str, str]) -> str | None:
    """The dotted import path a call's callee resolves to (`dt.now` -> `datetime.datetime.now`), or `None`."""
    attrs: list[str] = []
    while isinstance(func, ast.Attribute):
        attrs.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name):
        return None
    if func.id in imports:
        root = imports[func.id]
    else:
        obj = getattr(fn, "__globals__", {}).get(func.id)
        if isinstance(obj, ModuleType):
            root = obj.__name__
        else:
            # A bound method of a hidden instance (`from random import random`) has no `__module__` of its own.
            module = getattr(obj, "__module__", None) or type(getattr(obj, "__self__", None)).__module__
            name = getattr(obj, "__name__", None)
            if name is None:
                return None
            root = f"{module}.{name}"
    return ".".join([root.replace("posix.", "os.", 1), *reversed(attrs)])


class Impure(Check):
    """Shared-state writes (`global`, `nonlocal`) and mutable default arguments.

    Served steps run concurrently on one executable, so shared state in a step is a race.
    """

    type: Literal["impure"] = "impure"
    code = "DCD002"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node in ctx.calls():
            src = ctx.source(node)
            if src is None:
                continue
            path = node.origin.path
            for stmt in ast.walk(src.tree):
                if isinstance(stmt, (ast.Global, ast.Nonlocal)):
                    kind = "global" if isinstance(stmt, ast.Global) else "nonlocal"
                    what = "the global" if kind == "global" else "the closure cell"
                    for n in stmt.names:
                        yield ctx.finding(node, f"{path} writes {what} {n!r}; served steps run concurrently on one "
                                          "Executable, so concurrent calls race on it",
                                          fix="return the value instead", line=src.line(stmt), kind=kind, name=n)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a = stmt.args
                    pairs = list(zip(a.args[len(a.args) - len(a.defaults):], a.defaults))
                    pairs += [(arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
                    for arg, default in pairs:
                        if _mutable_default(default):
                            yield ctx.finding(node, f"{path} has a mutable default for {arg.arg!r}; its state persists "
                                              "across calls, and served steps run concurrently on one Executable",
                                              fix="default to None and build it in the body",
                                              line=src.line(default), kind="mutable_default", arg=arg.arg)


def _mutable_default(default: ast.expr) -> bool:
    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
        return True
    return (isinstance(default, ast.Call) and isinstance(default.func, ast.Name)
            and default.func.id in _MUTABLE_BUILTINS)


class MutatesInput(Check):
    """A step that changes an argument in place (`.append`, `x[k] = ...`, `inplace=True`).

    The argument is the caller's own object, and served steps run concurrently
    on one Executable, so another call or step can see the change.
    """

    type: Literal["mutates_input"] = "mutates_input"
    code = "DCD003"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node in ctx.calls():
            src = ctx.source(node)
            fn = src.tree.body[0] if src and isinstance(src.tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)) else None
            if fn is None:
                continue
            a = fn.args
            # ponytail: a name rebound anywhere in the body (`xs = list(xs)`) counts as a copy; flow-sensitive tracking if that misses
            rebound = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            names = {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs]} - rebound
            for stmt in ast.walk(fn):
                hit = _mutated(stmt, names)
                if hit:
                    arg, how = hit
                    yield ctx.finding(node, f"{node.origin.path} changes its argument {arg!r} in place ({how}); served "
                                      "steps run concurrently on one Executable, so other calls can see the change",
                                      fix="copy it first, or build and return a new value",
                                      line=src.line(stmt), arg=arg, how=how)


def _root(expr: ast.expr) -> str | None:
    while isinstance(expr, (ast.Subscript, ast.Attribute)):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def _mutated(stmt: ast.AST, names: set[str]) -> tuple[str, str] | None:
    if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute) and (arg := _root(stmt.func.value)) in names:
        if any(k.arg == "inplace" and isinstance(k.value, ast.Constant) and k.value.value is True for k in stmt.keywords):
            return arg, f".{stmt.func.attr}(inplace=True)"
        if stmt.func.attr in _MUTATING_METHODS:
            return arg, f".{stmt.func.attr}()"
    targets = (stmt.targets if isinstance(stmt, (ast.Assign, ast.Delete)) else
               [stmt.target] if isinstance(stmt, (ast.AugAssign, ast.AnnAssign)) else [])
    for t in targets:
        if isinstance(t, (ast.Subscript, ast.Attribute)) and (arg := _root(t)) in names:
            return arg, "item assignment" if isinstance(t, ast.Subscript) else "attribute assignment"
    return None
