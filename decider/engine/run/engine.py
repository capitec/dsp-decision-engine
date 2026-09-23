from __future__ import annotations

from typing import Any, Literal, Mapping

import polars as pl

from decider.engine.params import NodeParams, ParamsCache
from decider.engine.run.params import RunParams, RunReport, check_namespaces
from decider.engine.run.runners.base import Runner
from decider.engine.run.runners.interpreted import InterpretedRunner
from decider.engine.run.state import State
from decider.engine.wiring import Plan, resolve

# Mode name -> runner class; a new mode is one entry here.
RUNNERS: dict[str, type] = {"interpreted": InterpretedRunner}


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
            raise ValueError(f"params_validation must be 'eager' or 'lazy', not {params_validation!r}")
        self.params_validation = params_validation

    def bind(self, step_or_ir: Any, mode: str = "interpreted") -> Executable:
        """An `Executable` running `step_or_ir` in `mode` (only `"interpreted"` so far)."""
        runner = RUNNERS.get(mode)
        if runner is None:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(RUNNERS)}")
        return Executable(resolve(step_or_ir), runner(), self.params_validation == "lazy")


class Executable:
    """A bound pipeline: `run` a frame, or `score` one record.

    `report` describes the params validation of the latest call.

    Example::

        exe = Engine().bind(pipeline)
        exe.run(df)                       # a polars DataFrame
        exe.score({"net_income": 4100.0, ...})   # a dict
    """

    def __init__(self, plan: Plan, runner: Runner, lazy: bool):
        self.plan = plan
        self.runner = runner
        self.lazy = lazy
        self.nodes = {c.id: NodeParams(c.node.origin.path, c.node.params) for c in plan.calls if c.node.params}
        self.cache = ParamsCache()
        self._checked: set[str] = set()
        # ponytail: the latest call's report only; return it per call if concurrent callers need their own.
        self.report = RunReport()
        written = {o.name for c in plan.calls for o in c.node.outputs or ()}
        self._produced = written - {i.name for i in plan.inputs}

    def prepare(self, df: pl.DataFrame, params: Mapping[str, Any] | None = None,
                n: int | None = None) -> tuple[State, RunParams]:
        """The state and params a runner starts from; eager validation happens here.

        Example::

            state, params = exe.prepare(df)
            for checkpoint in exe.runner.iterate(exe.plan, state, params):
                ...
        """
        shadowed = sorted(self._produced & set(df.columns))
        if shadowed:
            name = shadowed[0]
            raise ValueError(
                f"'{name}' is produced by this pipeline and is also a column of the input frame. "
                f"Drop or rename the frame column '{name}', or relabel the step that writes it "
                f"(`.relabel(writes={{'{name}': '{name}_2'}})`), so neither silently shadows the other."
            )
        state = State.from_frame(self.plan, df, n)
        run = RunParams(self.nodes, {} if params is None else params, self.cache)
        self.report = run.report
        if run.key not in self._checked:
            check_namespaces(run.doc, self.nodes)
            if not self.lazy:
                for call_id in self.nodes:
                    run.bundle(call_id, state.n)
            self._checked.add(run.key)
        return state, run

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

        Example::

            exe.score({"net_income": 4100.0, "expenses": 1500.0})["disposable_income"]  # 2600.0
        """
        state, run = self.prepare(pl.DataFrame([dict(record)]), params, n=1)
        for _ in self.runner.iterate(self.plan, state, run):
            pass
        return self.output(state).row(0, named=True)

    def output(self, state: State) -> pl.DataFrame:
        """The output frame of a finished run: unread frame columns, then every produced output."""
        produced = {k: v for k, v in self.plan.outputs.items() if v.producer is not None}
        drops = set(self.plan.drops)
        kept = [state.frame[c] for c in state.frame.columns if c not in drops and c not in produced]
        return pl.DataFrame(kept + [state.column(k, v) for k, v in produced.items()])
