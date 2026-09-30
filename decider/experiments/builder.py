"""The Python facade over the experiment schema: a thin builder feeding `run_experiment`.

`Experiment` assembles the same `ExperimentDef` a YAML file holds, so the
builder and the asset tell one story; execution still happens only in
`run_experiment`.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from decider.experiments.model import ComparisonPolicy, ExperimentDef, FlowSpec, InputSpec, Scenario, Tolerance
from decider.experiments.runner import run_experiment
from decider.lifecycle import JobHandle


class Experiment:
    """A builder for one experiment: name, flow, input, scenarios, comparison, then `run`.

    Example::

        exp = (Experiment("drift", pipeline, df)
               .scenario("baseline")
               .scenario("limit_0.3", params={"demo": {"approved": {"limit": 0.3}}}))
        result = exp.run()
    """

    def __init__(self, name: str, step, frame: pl.DataFrame, *, params=None, revision: str = "HEAD",
                 entry: str = "", decider: str = "", input_path: str = "", id_column: str | None = None):
        self.step = step
        self.frame = frame
        self.params = params if params is not None else step.parameters().defaults()
        self.def_ = ExperimentDef(
            name=name,
            flow=FlowSpec(entry=entry, decider=decider),
            revision=revision,
            input=InputSpec(path=input_path, id_column=id_column),
        )

    def scenario(self, name: str, *, params: dict[str, Any] | None = None,
                 overrides: dict[str, Any] | None = None, at: str | None = None,
                 revision: str | None = None, row: int | None = None) -> "Experiment":
        self.def_.scenarios.append(Scenario(name=name, params=params or {}, overrides=overrides or {},
                                            at=at, revision=revision, row=row))
        return self

    def compare(self, *, baseline: str = "baseline", outputs: tuple[str, ...] = (),
                rtol: float = 1e-6, atol: float = 1e-9, determinism: bool = True) -> "Experiment":
        self.def_.comparison = ComparisonPolicy(
            baseline=baseline, outputs=outputs, tolerance=Tolerance(rtol=rtol, atol=atol),
            determinism=determinism)
        return self

    def run(self, *, job: JobHandle | None = None, resume=None, out_dir: str | None = None):
        return run_experiment(self.def_, self.step, self.frame, self.params, job=job, resume=resume, out_dir=out_dir)
