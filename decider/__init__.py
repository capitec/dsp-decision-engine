"""decider: decision pipelines from plain Python functions, run over polars frames or single records.

Start here: `decider guide` prints the getting-started guide (`decider/GUIDE.md`
in the installed package). Don't read the repo's `docs/` folder: it describes
an older version. The guide and the docstrings are current.

Quickstart::

    import polars as pl
    from decider import flow, param

    def ratio(income: float, debt: float) -> float:                  # plain arguments: request data
        return debt / income

    def affordable(ratio: float, limit: float = param(0.4)) -> bool:  # param(): policy, from params.json
        return ratio <= limit

    pipeline = flow(ratio, affordable, name="afford")
    df = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})
    pipeline.run(df)                                          # income, debt, affordable
    pipeline.compile("fused").score({"income": 1000.0, "debt": 200.0})   # compile once, then one record at a time (a dict)

A function's arguments are the columns it reads and its name is the column it
writes. `flow` runs steps in written order (a later write wins), `dag` in
dependency order; `branch`, `loop` and `ConfigurableStep` subclasses (trees,
tables, scorecards loaded from JSON) compose the same way. Tunable values are
`param()`s, retuned per run through a params document without rebuilding;
rows a step loops over (rate ladders, caps) are a `param_table()`.
Request data is never a `param()`.

More::

    pipeline.emit("ratio").run(df)                            # intermediates are dropped unless emitted
    pipeline.run(df, params={"afford": {"affordable": {"limit": 0.9}}})
    pipeline.parameters().defaults()                          # the params document to edit
    s = pipeline.session(df); s.break_at("afford/affordable"); s.resume()   # debug step by step

`decider template NAME` writes a starter project; `decider build` and
`decider serve` build and serve it. Companion packages, one per task:

- `decider.ops`: the value types a step declares (`Text`, `Currency`, `Columnar`, `Percent`, ...)
- `decider.lab`: work on a flow away from serving: `data`, `checks`, `diff`, `testing`, `experiments`
- `decider.serving`: `RequestHandler` and the HTTP app
- `decider.config`: `JsonFileStore`, versioned params documents
- `decider.debug`: `Session` and the events it emits
- `decider.trace`: `TraceSink`, per-record decision traces
- `decider.ir`: the intermediate representation, for step-type authors

Tree, table and scorecard documents are `TreeConfig`, `DecisionTableConfig` and
`ScorecardConfig`, loaded on first use. Build, wiring, params and missing-input
errors are `DeciderError`s (`decider.exceptions` has the subclasses).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from decider._engine.params import ParamTable, missing_as, param, param_table
from decider.lab import describe
from decider.exceptions import DeciderError, FallbackWarning
from decider.steps import (ConfigurableStep, EachMode, ParamRef, Step, StepRef, Value, allow_fallback, branch, dag,
                           each, flow, frame_step, helper, loop, optimise, step)
from decider.steps.flowfile import load_flow

if TYPE_CHECKING:
    from decider._engine.run import Engine, Executable
    from decider.steps.scorecard import ScorecardConfig
    from decider.steps.tables import DecisionTableConfig
    from decider.steps.trees import TreeConfig

__all__ = [
    # build
    "step", "frame_step", "flow", "dag", "branch", "loop", "each", "optimise", "EachMode", "allow_fallback",
    "helper", "param", "param_table", "missing_as", "ParamTable", "TreeConfig", "DecisionTableConfig", "ScorecardConfig",
    # run
    "Engine", "Executable", "load_flow", "describe",
    # extend
    "Step", "ConfigurableStep", "Value", "ParamRef", "StepRef",
    # errors
    "DeciderError", "FallbackWarning",
]

# Names whose modules import numba load on first use, so `import decider` stays light.
_LAZY = {
    "Engine": "decider._engine.run",
    "Executable": "decider._engine.run",
    "TreeConfig": "decider.steps.trees",
    "DecisionTableConfig": "decider.steps.tables",
    "ScorecardConfig": "decider.steps.scorecard",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'decider' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
