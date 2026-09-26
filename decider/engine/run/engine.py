from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np
import polars as pl

from decider.engine.params import NodeParams, ParamsCache
from decider.engine.run.params import RunParams, RunReport, check_namespaces
from decider.engine.run.runners.base import Runner
from decider.engine.run.runners.fused import FusedRunner
from decider.engine.run.runners.interpreted import InterpretedRunner
from decider.engine.run.runners.stepped import SteppedRunner
from decider.engine.ir.decls import base_annotation
from decider.engine.run.state import State, dtype_of
from decider.engine.wiring import Plan, resolve
from decider.engine.wiring.plan import Version
from decider.exceptions import EngineError

if TYPE_CHECKING:
    from decider.engine.debug import Session

Mode = Literal["interpreted", "stepped", "fused"]

# Mode name -> runner class; a new mode is one entry here.
RUNNERS: dict[str, type] = {"interpreted": InterpretedRunner, "stepped": SteppedRunner, "fused": FusedRunner}

_NO_FRAME = pl.DataFrame()


class Engine:
    """Turns a step (or its IR) into something that runs.

    Args:
        params_validation: `"eager"` validates every node's params when a
            params document is first seen and raises on any invalid one;
            `"lazy"` validates a node the first time it runs, so an invalid
            param in a node no row reaches never fails a run.

    Example::

        exe = Engine(params_validation="lazy").bind(pipeline)
        out = exe.run(df, params={"term": {"cap_by_income": {"cap": 36.0}}})
    """

    def __init__(self, params_validation: Literal["eager", "lazy"] = "eager"):
        if params_validation not in ("eager", "lazy"):
            raise EngineError(f"params_validation must be 'eager' or 'lazy', not {params_validation!r}")
        self.params_validation = params_validation

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
        return Executable(resolve(step_or_ir), runner(), self.params_validation == "lazy", step_or_ir)


class Executable:
    """A bound pipeline: `run` a frame, or `score` one record.

    `report` describes the params validation of the latest call; `step` is
    what it was bound from (a step, a function or an IR).

    Example::

        exe = Engine().bind(pipeline)
        exe.run(df)                       # a polars DataFrame
        exe.score({"net_income": 4100.0, ...})   # a dict
    """

    def __init__(self, plan: Plan, runner: Runner, lazy: bool, step: Any = None):
        self.plan = plan
        self.step = step
        self.runner = runner
        self.lazy = lazy
        self.nodes = {c.id: NodeParams(c.node.origin.path, c.node.params) for c in plan.calls if c.node.params}
        self.cache = ParamsCache()
        self._checked: set[str] = set()
        # ponytail: the latest call's report only; return it per call if concurrent callers need their own.
        self.report = RunReport()
        written = {o.name for c in plan.calls for o in c.node.outputs or ()}
        self._produced = written - {i.name for i in plan.inputs}
        inputs: dict[np.dtype, list[Version]] = {}
        for v in plan.versions:
            if v.producer is None:
                inputs.setdefault(dtype_of(base_annotation(v.annotation)), []).append(v)
        self._inputs = list(inputs.items())
        self._results = [(k, v) for k, v in plan.outputs.items() if v.producer is not None]
        self._hidden = set(plan.drops) | {k for k, _ in self._results}
        self._record_path = not any(c.node.kind == "frame" for c in plan.calls)

    def prepare(self, df: pl.DataFrame, params: Mapping[str, Any] | None = None,
                n: int | None = None) -> tuple[State, RunParams]:
        """The state and params a runner starts from; eager validation happens here.

        Example::

            state, params = exe.prepare(df)
            for checkpoint in exe.runner.iterate(exe.plan, state, params):
                ...
        """
        self._check_shadowing(df.columns)
        state = State.from_frame(self.plan, df, n)
        return state, self._params(params, state.n)

    def _check_shadowing(self, columns) -> None:
        shadowed = sorted(self._produced.intersection(columns))
        if shadowed:
            name = shadowed[0]
            raise EngineError(
                f"'{name}' is produced by this pipeline and is also a column of the input frame. "
                f"Drop or rename the frame column '{name}', or relabel the step that writes it "
                f"(`.relabel(writes={{'{name}': '{name}_2'}})`), so neither silently shadows the other."
            )

    def _params(self, params: Mapping[str, Any] | None, n: int) -> RunParams:
        run = RunParams(self.nodes, {} if params is None else params, self.cache, self.lazy)
        self.report = run.report
        if run.key not in self._checked:
            check_namespaces(run.doc, self.nodes)
            if not self.lazy:
                for call_id in self.nodes:
                    run.bundle(call_id, n)
            self._checked.add(run.key)
        return run

    def run(self, df: pl.DataFrame, params: Mapping[str, Any] | None = None) -> pl.DataFrame:
        """Run over a frame: the input columns plus every output, minus drops.

        Example::

            out = exe.run(df, params={"shared": {"min_ratio": 0.4}})
        """
        state, run = self.prepare(df, params)
        for _ in self.runner.iterate(self.plan, state, run):
            pass
        return self.output(state)

    def score(self, record: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Run one record, given as a dict of input values; returns a dict of the output row.

        Uses the same kernels as `run`. A pipeline without frame steps never
        builds a polars frame on this path.

        Example::

            exe.score({"net_income": 4100.0, "expenses": 1500.0})["disposable_income"]  # 2600.0
        """
        if not self._record_path:
            state, run = self.prepare(pl.DataFrame([dict(record)]), params, n=1)
            for _ in self.runner.iterate(self.plan, state, run):
                pass
            return self.output(state).row(0, named=True)
        # A pipeline without frame steps never needs a frame: the record goes
        # straight into one-row arrays and the result straight into a dict.
        self._check_shadowing(record.keys())
        state = State(self.plan, _NO_FRAME, 1)
        for dtype, versions in self._inputs:
            _load(state, record, versions, dtype)
        run = self._params(params, 1)
        for _ in self.runner.iterate(self.plan, state, run):
            pass
        out = {k: x for k, x in record.items() if k not in self._hidden}
        for k, v in self._results:
            values, valid = state.read(v)
            out[k] = None if valid is not None and not valid[0] else values.tolist()[0]
        return out

    def session(self, df: pl.DataFrame, params: Mapping[str, Any] | None = None) -> Session:
        """A debug `Session` over `df`, paused before anything runs.

        Example::

            s = exe.session(df)
            s.break_at("term")
            s.resume()
        """
        from decider.engine.debug import Session

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


def _load(state: State, record: Mapping[str, Any], versions: list[Version], dtype: np.dtype) -> None:
    # One array per dtype, viewed per input: far cheaper than one array per input.
    given = [v for v in versions if v.name in record]
    if not given:
        return
    xs = [record[v.name] for v in given]
    nulls = [x is None for x in xs]
    if dtype == object:
        block = np.empty(len(xs), object)
        # One by one, so a list or tuple value stays one element.
        for k, x in enumerate(xs):
            block[k] = x
    else:
        block = np.array([0 if null else x for x, null in zip(xs, nulls)] if any(nulls) else xs, dtype)
        # Read-only like a column read from a frame, so one kernel specialisation serves both.
        block.flags.writeable = False
    for k, (v, null) in enumerate(zip(given, nulls)):
        state.write(v, block[k:k + 1], valid=np.zeros(1, bool) if null else None)
