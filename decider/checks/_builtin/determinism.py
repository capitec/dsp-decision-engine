from __future__ import annotations

import ast
from typing import Iterator, Literal

from decider.checks._base import Check, CheckContext
from decider.checks._report import Finding

__all__ = ["Impure", "WallClock"]

_WALL_CLOCK: dict[str, set[str]] = {
    "time": {"time", "time_ns", "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns",
             "process_time", "process_time_ns", "thread_time", "thread_time_ns", "clock"},
    "datetime": {"now", "utcnow", "today"},
    "random": {"random", "randint", "randrange", "uniform", "choice", "choices", "shuffle",
               "sample", "gauss", "randbytes"},
    "uuid": {"uuid1", "uuid4"},
    "secrets": {"token_bytes", "token_hex", "token_urlsafe", "randbelow", "choice"},
    "os": {"urandom"},
}

_MUTABLE_BUILTINS = {"list", "dict", "set"}


class WallClock(Check):
    """Wall-clock and randomness reads (`time.time()`, `datetime.now()`, `random.random()`) make a flow non-deterministic."""

    type: Literal["wall_clock"] = "wall_clock"
    code = "DCD001"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node in ctx.calls():
            src = ctx.source(node)
            for call in ast.walk(src.tree) if src else ():
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                    continue
                module = getattr(call.func.value, "id", None)
                if module in _WALL_CLOCK and call.func.attr in _WALL_CLOCK[module]:
                    text = f"{module}.{call.func.attr}()"
                    yield ctx.finding(node, f"{node.origin.path} reads {text}; a wall-clock read makes the flow "
                                      "non-deterministic", fix="pass the value in as an input column",
                                      line=src.line(call), call=text)


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
                        yield ctx.finding(node, f"{path} writes {what} {n!r}; concurrent calls race on it",
                                          fix="return the value instead", line=src.line(stmt), kind=kind, name=n)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = stmt.args.args[len(stmt.args.args) - len(stmt.args.defaults):]
                    for arg, default in zip(args, stmt.args.defaults):
                        if _mutable_default(default):
                            yield ctx.finding(node, f"{path} has a mutable default for {arg.arg!r}; "
                                              "its state persists across calls", fix="default to None and build it in the body",
                                              line=src.line(default), kind="mutable_default", arg=arg.arg)


def _mutable_default(default: ast.expr) -> bool:
    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
        return True
    return (isinstance(default, ast.Call) and isinstance(default.func, ast.Name)
            and default.func.id in _MUTABLE_BUILTINS)
