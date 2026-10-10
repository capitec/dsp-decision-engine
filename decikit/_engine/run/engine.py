from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

from decikit._engine.concurrency import Cache, once
from decikit._engine.params import NodeParams, ParamsCache
from decikit._engine.params.shadow import warn_shadowed_tables
from decikit._engine.run.hidden import warn_hidden_state
from decikit._engine.run.params import RunParams, RunReport, check_namespaces
from decikit._engine.run.representations import despan
from decikit._engine.run.runners.base import CompiledRunner, Runner
from decikit._engine.run.runners.fused import FusedRunner
from decikit._engine.run.runners.interpreted import InterpretedRunner
from decikit._engine.run.runners.stepped import SteppedRunner
from decikit._engine.ir.decls import ParamDecl, base_annotation
from decikit._engine.run.state import State, declared_dtype, dtype_of, load_record, record_value
from decikit._engine.trace.envelope import StepTable
from decikit._engine.types import decimal_boxer, decimal_info
from decikit._engine.wiring import Plan, resolve
from decikit._engine.wiring.plan import Branch, Call, Loop, ScatterGather, Sequence, NodeVersion
from decikit.exceptions import EngineError
from decikit.ops.nested import struct_item

if TYPE_CHECKING:
    from decikit._engine.debug import Session

Mode = Literal["interpreted", "stepped", "fused"]

# Mode name -> runner class; a new mode is one entry here.
RUNNERS: dict[str, type] = {"interpreted": InterpretedRunner, "stepped": SteppedRunner, "fused": FusedRunner}
_QUIET = (InterpretedRunner.iterate, SteppedRunner.iterate)

_NO_FRAME = pl.DataFrame()


class Engine:
    """Turns a step (or its IR) into something that runs.

    Args:
        params_validation: `"eager"` validates every node's params when a
            params document is first seen and raises on any invalid one;
            `"lazy"` validates a node the first time it runs, so an invalid
            param in a node no row reaches never fails a run.
        strict_compile: in `"stepped"` and `"fused"` modes, a step that would
            run in Python raises at the first run instead of warning. Mark the
            ones you accept `@allow_fallback`.

    Example::

        exe = Engine(params_validation="lazy").bind(pipeline)
        out = exe.run(df, params={"term": {"cap_by_income": {"cap": 36.0}}})
    """

    def __init__(self, params_validation: Literal["eager", "lazy"] = "eager", strict_compile: bool = False):
        if params_validation not in ("eager", "lazy"):
            raise EngineError(f"params_validation must be 'eager' or 'lazy', not {params_validation!r}")
        self.params_validation = params_validation
        self.strict_compile = strict_compile

    def bind(self, step_or_ir: Any, mode: Mode = "interpreted") -> Executable:
        """An `Executable` running `step_or_ir` in `mode`.

        Args:
            mode: `"interpreted"` (plain Python, pauses at every node and
                inside trees), `"stepped"` (one numba kernel per node, pauses
                at every node) or `"fused"` (one kernel per innermost sequence
                of scalar steps, pauses at kernel boundaries). All three give
                the same answers.

        Example::

            exe = Engine().bind(pipeline, mode="fused")
        """
        runner = RUNNERS.get(mode)
        if runner is None:
            raise EngineError(f"unknown mode {mode!r}; expected one of {sorted(RUNNERS)}")
        runner = runner() if runner is InterpretedRunner else runner(strict=self.strict_compile)
        exe = Executable(resolve(step_or_ir), runner, self.params_validation == "lazy", step_or_ir)
        warn_hidden_state(exe.plan, strict=self.strict_compile)
        return exe


class Executable:
    """A bound pipeline: `run` a polars frame or one record (a dict).

    `report` describes the params validation of the latest call on the
    calling thread; `step` is what it was bound from (a step, a function or
    an IR). Safe to call from many threads at once.

    Example::

        exe = Engine().bind(pipeline)
        exe.run(df)                       # a polars DataFrame
        exe.run({"net_income": 4100.0, ...})   # a dict
    """

    def __init__(self, plan: Plan, runner: Runner, lazy: bool, step: Any = None):
        self.plan = plan
        self.step = step
        self.runner = runner
        self.lazy = lazy
        self.nodes = {c.id: NodeParams(c.node.origin.path, c.node.params) for c in plan.calls if c.node.params}
        # A scatter-gather node's hoisted child params validate and bundle like a call's, under its path.
        for sg in _scatter_gathers(plan.root):
            if sg.node.params:
                self.nodes[sg.node.origin.path] = NodeParams(sg.node.origin.path, sg.node.params)
        self.cache = ParamsCache()
        self._checked = Cache()
        self._latest = threading.local()
        written = {o.name for c in plan.calls for o in c.node.outputs or ()}
        self._produced = written - {i.name for i in plan.inputs}
        self._params_at = {d.shared_key or d.name: _where(n.path, d) for n in self.nodes.values() for d in n.decls}
        for i in plan.inputs:
            self._params_at.pop(i.name, None)
        # One set test on the hot path covers both checks; nothing else happens unless it hits.
        self._watched = self._produced | self._params_at.keys()
        self._warned = Cache()
        self._declared = {i.name: _declared(i.annotation) for i in plan.inputs}
        inputs: dict[np.dtype, list[NodeVersion]] = {}
        for v in plan.versions:
            if v.producer is None:
                inputs.setdefault(dtype_of(base_annotation(v.annotation)), []).append(v)
        self._inputs = list(inputs.items())
        self._results = [(k, v) for k, v in plan.outputs.items() if v.producer is not None]
        # A Decimal output is stored as its int64 and handed back as a Decimal.
        self._boxers = {k: box for k, v in self._results if (box := decimal_boxer(v.annotation)) is not None}
        # Only a struct output's annotation shapes its record; looking up any other per call hashes it,
        # and hashing a `Literal` of thousands of choices cost more than the rest of a record's run.
        self._structs = {k: v.annotation for k, v in self._results if struct_item(base_annotation(v.annotation))}
        self._hidden = set(plan.drops) | {k for k, _ in self._results}
        self._record_path = not any(c.node.kind == "frame" for c in plan.calls)
        # The trace's per-run constant table: step ref (call index + 1) -> durable origin.
        self._trace_table = StepTable(tuple(c.node.origin for c in plan.calls))

    def prepare(self, df: pl.DataFrame, params: Mapping[str, Any] | None = None,
                n: int | None = None) -> tuple[State, RunParams]:
        """The state and params a runner starts from; eager validation happens here.

        Example::

            state, params = exe.prepare(df)
            for checkpoint in exe.runner.iterate(exe.plan, state, params):
                ...
        """
        self._check_shadowing(df.columns)
        # A column that is all null, or all empty lists, infers a `Null` dtype, which the Arrow import rejects.
        fix = {c: self._declared.get(c) or k for c, t in df.schema.items() if (k := _concrete(t)) != t}
        if fix:
            df = df.with_columns(pl.col(c).cast(t) for c, t in fix.items())
        state = State.from_frame(self.plan, df, n)
        return state, self._params(params, state.n)

    def _check_shadowing(self, columns) -> None:
        hit = self._watched.intersection(columns)
        if not hit:
            return
        for name in sorted(hit - self._produced - self._warned.keys()):
            once(self._warned, name, lambda: self._warn_param(name))
        shadowed = sorted(hit & self._produced)
        if shadowed:
            name = shadowed[0]
            raise EngineError(
                f"'{name}' is produced by this pipeline and is also a column of the input frame. "
                f"Drop or rename the frame column '{name}', or relabel the step that writes it "
                f"(`.relabel(writes={{'{name}': '{name}_2'}})`), so neither silently shadows the other."
            )

    def _warn_param(self, name: str) -> bool:
        warnings.warn(f"'{name}' is a param, not an input, so its value in the record or frame is ignored; "
                      f"pass it in the params document instead: {self._params_at[name]}", stacklevel=5)
        return True

    @property
    def report(self) -> RunReport:
        return getattr(self._latest, "report", None) or RunReport()

    def _params(self, params: Mapping[str, Any] | None, n: int) -> RunParams:
        run = RunParams(self.nodes, {} if params is None else params, self.cache, self.lazy)
        self._latest.report = run.report
        if run.key not in self._checked:
            self._check_once(run, n)
        return run

    def _check_once(self, run: RunParams, n: int) -> None:
        once(self._checked, run.key, lambda: self._check(run, n))

    def _check(self, run: RunParams, n: int) -> bool:
        check_namespaces(run.doc, self.nodes)
        if not self.lazy:
            for call_id in self.nodes:
                run.bundle(call_id, n)
        return True

    def run(self, data: Any, params: Mapping[str, Any] | None = None, trace=None) -> pl.DataFrame | dict[str, Any]:
        """Run a polars frame, or one record given as a dict of input values.

        A `DataFrame` returns a `DataFrame`: the input columns plus every
        output, minus drops. A `LazyFrame` is collected first. A dict returns
        a dict of the output row, through the same kernels; a pipeline without
        frame steps never builds a polars frame for it, so this is the fast
        path for one record.

        `params` is treated as immutable: pass the same document object call
        after call and it is never hashed or validated again, so a value
        edited in place is not seen. To change a value, pass a new dict
        (`copy.deepcopy(doc)`, then edit the copy). `trace` is an optional
        `TraceSink`: when it is not given, nothing is captured (no-op).

        Example::

            out = exe.run(df, params={"shared": {"min_ratio": 0.4}})
            exe.run({"net_income": 4100.0, "expenses": 1500.0})["disposable_income"]  # 2600.0
        """
        # A dict first: it is the serving hot path, the `Mapping` check is slow, and a dict is never a frame.
        if isinstance(data, dict):
            return self._run_record(data, params, trace)
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        if isinstance(data, pl.DataFrame):
            state, run = self.prepare(data, params)
            self._iterate(state, run, trace)
            return self.output(state)
        if isinstance(data, Mapping):
            return self._run_record(data, params, trace)
        raise TypeError(f"run() takes a dict record, a polars DataFrame or LazyFrame, not {type(data).__name__}")

    def _run_record(self, record: Mapping[str, Any], params: Mapping[str, Any] | None = None,
                    trace=None) -> dict[str, Any]:
        if not self._record_path:
            state, run = self.prepare(pl.DataFrame([dict(record)]), params, n=1)
            self._iterate(state, run, trace)
            return self.output(state).row(0, named=True)
        # A pipeline without frame steps never needs a frame: the record goes
        # straight into one-row arrays and the result straight into a dict.
        self._check_shadowing(record.keys())
        state = State(self.plan, _NO_FRAME, 1)
        for dtype, versions in self._inputs:
            load_record(state, record, versions, dtype)
        run = self._params(params, 1)
        self._iterate(state, run, trace)
        out = {k: x for k, x in record.items() if k not in self._hidden}
        for k, v in self._results:
            values, valid = state.read(v)
            if valid is not None and not valid[0]:
                out[k] = None
            elif (box := self._boxers.get(k)) is not None:
                out[k] = box(values.tolist()[0])
            elif (annotation := self._structs.get(k)) is not None:
                out[k] = record_value(values.tolist()[0], annotation)
            else:
                out[k] = despan(values.tolist()[0])
        return out

    def _iterate(self, state: State, run: RunParams, trace) -> None:
        table = self._trace_table if trace is not None else None
        # Nobody steps through this run, so the built-in runners skip building the checkpoints. A subclass that
        # overrides `iterate` may read them, and may not take `quiet`.
        quiet = getattr(type(self.runner), "iterate", None) in _QUIET
        for _ in (self.runner.iterate(self.plan, state, run, trace, table, quiet=True) if quiet
                  else self.runner.iterate(self.plan, state, run, trace, table)):
            pass

    def fallbacks(self) -> dict[str, str]:
        """Every step that runs outside the shared kernel, keyed by step path, with the reason why.

        A step the author accepted with `@allow_fallback` is reported too, with
        `"@allow_fallback: "` in front of its reason.

        Example::

            exe.fallbacks()   # {"order/order_total": "reads 'items' as list[dict], which no kernel takes"}
        """
        if not isinstance(self.runner, CompiledRunner):
            return {}
        if self.runner._plan is not self.plan:
            self.runner._compile(self.plan, self.lazy)
        return self.runner.fallbacks()

    def warn_shadowed_tables(self, params: Mapping[str, Any] | None) -> None:
        """Warn with `ShadowedTableWarning` for each `seeded_from` table that the params document `params` sets.

        `RequestHandler.stage()` calls this for the version it stages.

        Example::

            exe.warn_shadowed_tables({"pricing": {"priced": {"ladder": [{"floor": 0, "rate": 0.3}]}}})
        """
        warn_shadowed_tables(self.nodes.values(), params)

    def session(self, df: pl.DataFrame, params: Mapping[str, Any] | None = None) -> Session:
        """A debug `Session` over `df`, paused before anything runs.

        Example::

            s = exe.session(df)
            s.break_at("term")
            s.resume()
        """
        from decikit._engine.debug import Session

        return Session(self, df, params)

    def output(self, state: State) -> pl.DataFrame:
        """The output frame of a finished run: unread frame columns, then every produced output."""
        frame = state.frame
        results = [state.column(k, v) for k, v in self._results]
        names = frame.columns
        if names and self._hidden.isdisjoint(names):
            # Every frame column passes through: hstack is several times cheaper than a new frame.
            return frame.hstack(results)
        return pl.DataFrame([frame.get_column(c) for c in names if c not in self._hidden] + results)


def _declared(annotation: Any) -> pl.DataType | None:
    info = decimal_info(annotation)
    return pl.Decimal(38, info.scale) if info is not None else declared_dtype(annotation)


def _scatter_gathers(r) -> Any:
    if isinstance(r, ScatterGather):
        yield r
        yield from _scatter_gathers(r.body)
        if r.accumulate is not None:
            yield from _scatter_gathers(r.accumulate)
    elif isinstance(r, Sequence):
        for c in r.children:
            yield from _scatter_gathers(c)
    elif isinstance(r, Branch):
        yield from _scatter_gathers(r.condition)
        for a in r.arms:
            yield from _scatter_gathers(a)
    elif isinstance(r, Loop):
        yield from _scatter_gathers(r.condition)
        yield from _scatter_gathers(r.body)


def _where(path: str, decl: ParamDecl) -> str:
    # The params-document entry that sets `decl`, as JSON.
    if decl.shared_key is not None:
        return json.dumps({"shared": {decl.shared_key: "..."}})
    doc: Any = {decl.name: "..."}
    for part in reversed(path.split("/")):
        doc = {part: doc}
    return json.dumps(doc)


def _concrete(dtype: pl.DataType) -> pl.DataType:
    # `dtype` with every `Null` inside it made `String`: any concrete type holds all-null values.
    # A `Struct` field is left alone; a null field is still a field, not an empty column, and a
    # reader falls back to Python rather than silently re-type it.
    if dtype == pl.Null:
        return pl.String
    if isinstance(dtype, pl.List):
        return pl.List(_concrete(dtype.inner))
    if isinstance(dtype, pl.Array):
        return pl.Array(_concrete(dtype.inner), dtype.size)
    return dtype
