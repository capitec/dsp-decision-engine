from __future__ import annotations

import importlib.metadata
import inspect
from abc import abstractmethod
from functools import cached_property
from typing import Any, Callable, ClassVar, Iterable, Literal

from pydantic import ConfigDict

from decider._engine.wiring import resolve
from decider.checks._ast import Source, parse
from decider.checks._report import Finding, Severity, Suppression
from decider.contract import StepRef, describe
from decider.ir import CallNode, IRNode, iter_nodes, step_map, to_ir
from decider.registry import BaseRegistryModule

# What `decider.checks.standard` runs: the static checks, which need no data and no execution.
_NUMERIC = ("int64_overflow", "float_money", "bare_round", "division_by_variable", "int_true_division",
            "pow_exp_overflow", "float_equality", "nan_propagation", "accumulator_dtype", "unsafe_cast")
STANDARD = ("wall_clock", "impure", "mutates_input", *_NUMERIC, "param_unconstrained", "param_default_out_of_bounds",
            "param_unused", "durable_ids", "duplicate_ids", "unknown_suppression")


class Check(BaseRegistryModule, root=True, group="checks"):
    """One diagnostic. Subclass it with a `type` alias; config names it by alias or `"module:Class"`.

    Set `code` (stable, never reused; a third-party check uses its own prefix),
    `needs_data` when it only runs given a frame, and `executes` when it runs
    the flow. `run` yields findings built with `ctx.finding`.

    Example::

        from typing import Literal
        from decider.checks import Check

        class NoPrint(Check):
            type: Literal["no_print"] = "no_print"
            code = "ACME002"

            def run(self, ctx):
                for node in ctx.calls():
                    ...
                    yield ctx.finding(node, "prints", fix="log instead")

        checks.run(pipeline, [checks.standard, NoPrint(severity="error")])
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ClassVar[str]
    needs_data: ClassVar[bool] = False
    executes: ClassVar[bool] = False
    severity: Severity = Severity.WARNING

    @abstractmethod
    def run(self, ctx: CheckContext) -> Iterable[Finding]: ...


Check.lazy({
    "wall_clock": "decider.checks._builtin.determinism", "impure": "decider.checks._builtin.determinism",
    **dict.fromkeys(_NUMERIC, "decider.checks._builtin.numeric"),
    "mutates_input": "decider.checks._builtin.determinism",
    "durable_ids": "decider.checks._builtin.ids", "unknown_suppression": "decider.checks._builtin.ids",
    "duplicate_ids": "decider.checks._builtin.ids",
    "param_unconstrained": "decider.checks._builtin.params", "param_default_out_of_bounds": "decider.checks._builtin.params",
    "param_unused": "decider.checks._builtin.params",
    "missing_required": "decider.checks._builtin.data", "invalid_params_doc": "decider.checks._builtin.params",
    "crash_probe": "decider.checks._builtin.runtime", "mode_divergence": "decider.checks._builtin.runtime",
})


class CheckContext:
    """What a check sees: the flow's IR, plan and description, the data and params if given.

    Built once per `checks.run` and shared; every attribute is computed on first use.
    `calls()` are the call nodes of authored code, `source(node)` its parsed source,
    and `finding(...)` builds a `Finding` with code, check, step reference, file and
    line filled in, marked suppressed when the node or an ancestor suppresses it.

    Example::

        for node in ctx.calls():
            src = ctx.source(node)
            yield ctx.finding(node, "message", fix="do this", line=src.line(expr), expression="a * b")
    """

    def __init__(self, pipeline: Any, params: Any = None, frame: Any = None) -> None:
        self.pipeline = pipeline
        self.params = params
        self.frame = frame
        self._current: Check | None = None
        # The node of the `ConfigurableStep` whose `check` hook is running.
        self.node: IRNode | None = None
        self._sources: dict[int, Source | None] = {}

    @cached_property
    def ir(self) -> IRNode:
        return to_ir(self.pipeline)

    @cached_property
    def plan(self):
        return resolve(self.ir)

    @cached_property
    def description(self):
        return describe(self.pipeline)

    @cached_property
    def steps(self) -> dict[str, Any]:
        """The step object behind each node path."""
        return step_map(self.pipeline)

    @cached_property
    def nodes(self) -> list[IRNode]:
        return list(iter_nodes(self.ir))

    @cached_property
    def _refs(self) -> dict[int, StepRef]:
        # Same order and shape as the nodes of `ir`, which `describe` walks the same way.
        return {id(n): r for n, r in zip(self.nodes, self.description.nodes)}

    @cached_property
    def _suppressions(self) -> dict[int, list[tuple[str, str]]]:
        found: dict[int, list[tuple[str, str]]] = {}

        def walk(node: IRNode, inherited: list[tuple[str, str]]) -> None:
            own = inherited + [(name, node.origin.path) for name in node.origin.suppress_checks]
            found[id(node)] = own
            for child in node.children():
                walk(child, own)

        walk(self.ir, [])
        return found

    def calls(self) -> list[CallNode]:
        """Call nodes of authored code; generated kernels from `decider.steps` are skipped."""
        return [n for n in self.nodes if isinstance(n, CallNode)
                and not (getattr(n.fn, "__module__", None) or "").startswith("decider.steps")]

    def source(self, node: CallNode) -> Source | None:
        """The parsed source of `node`'s function, or `None` when it isn't available."""
        key = id(node.fn)
        if key not in self._sources:
            self._sources[key] = parse(node.fn, {i.arg: i.annotation for i in node.inputs or ()})
        return self._sources[key]

    def finding(self, node: IRNode, message: str, fix: str | None = None, *, line: int | None = None,
                code: str | None = None, check: str | None = None, severity: Severity | None = None,
                **detail: Any) -> Finding:
        """A finding at `node`. `code`, `check` and `severity` default to the running check's."""
        current = self._current
        code = code or (current.code if current else "")
        check = check or (current.type if current else "")
        severity = severity or (current.severity if current else Severity.WARNING)
        file = None
        if isinstance(node, CallNode):
            try:
                file = inspect.getsourcefile(node.fn)
            except TypeError:
                pass
        suppressed = next((Suppression(path=path, name=name) for name, path in self._suppressions[id(node)]
                           if name in (code, check)), None)
        return Finding(code=code, check=check, severity=severity, message=message, step=self._refs[id(node)],
                       file=file, line=line, fix=fix, detail=detail, suppressed=suppressed)


def check(code: str, severity: Severity | str = Severity.WARNING, alias: str | None = None) -> Callable[[Callable], Check]:
    """Turn a function `f(ctx)` yielding findings into a check, for a one-off rule.

    The alias defaults to the function's name. The result is a `Check` instance,
    ready for a `checks.run` list.

    Example::

        @checks.check(code="ACME002", severity="warning")
        def no_print(ctx):
            for node in ctx.calls():
                yield ctx.finding(node, "prints")

        checks.run(pipeline, [checks.standard, no_print])
    """
    def wrap(fn: Callable) -> Check:
        tag = alias or fn.__name__
        body = {"__module__": fn.__module__, "__qualname__": fn.__qualname__, "__doc__": fn.__doc__,
                "__annotations__": {"type": Literal[tag]}, "type": tag, "code": code,
                "run": lambda self, ctx: fn(ctx)}
        return type(fn.__name__, (Check,), body)(severity=Severity(severity))

    return wrap


def registered() -> dict[str, type[Check]]:
    """Every registered check class by alias (or import path), installed packages' included."""
    for entry in importlib.metadata.entry_points(group="decider.checks"):
        entry.load()
    return {tag: Check.resolve(tag) for tag in Check.json_schema()}
