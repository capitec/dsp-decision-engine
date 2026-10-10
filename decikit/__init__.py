"""decikit: decision pipelines from plain Python functions, run over polars frames or single records.

Start here: `decikit guide` prints the getting-started guide (`decikit/GUIDE.md`
in the installed package). Don't read the repo's `docs/` folder: it describes
an older version. The guide and the docstrings are current.

Quickstart::

    import polars as pl
    from decikit import flow, param

    def ratio(income: float, debt: float) -> float:                  # plain arguments: request data
        return debt / income

    def affordable(ratio: float, limit: float = param(0.4)) -> bool:  # param(): policy, from params.json
        return ratio <= limit

    pipeline = flow(ratio, affordable, name="afford")
    df = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})
    pipeline.run(df)                                          # income, debt, affordable
    pipeline.compile("fused").run({"income": 1000.0, "debt": 200.0})     # compile once, then one record at a time (a dict)

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

`decikit template NAME` writes a starter project; `decikit build` and
`decikit serve` build and serve it. Companion packages, one per task:

- `decikit.ops`: the value types a step declares (`Text`, `Currency`, `Columnar`, `Percent`, ...)
- `decikit.lab`: work on a flow away from serving: `data`, `checks`, `diff`, `testing`, `experiments`
- `decikit.serving`: `RequestHandler` and the HTTP app
- `decikit.config`: `JsonFileStore`, versioned params documents
- `decikit.debug`: `Session` and the events it emits
- `decikit.trace`: `TraceSink`, per-record decision traces
- `decikit.ir`: the intermediate representation, for step-type authors

Tree, table and scorecard documents are `TreeConfig`, `DecisionTableConfig` and
`ScorecardConfig`, loaded on first use. Build, wiring, params and missing-input
errors are `DecikitError`s (`decikit.exceptions` has the subclasses).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version
from typing import TYPE_CHECKING, Any

from decikit._serde import PolarsSchema
from decikit._engine.params import ParamTable, missing_as, param, param_table
from decikit.lab import describe
from decikit.exceptions import DecikitError, FallbackWarning, HiddenStateWarning, ShadowedTableWarning
from decikit.steps import (ConfigurableStep, CustomFunction, EachMode, ParamRef, Step, StepRef, Value, allow_fallback, branch, dag,
                           each, flow, frame_step, helper, loop, optimise, step)
from decikit.steps.flowfile import load_flow

if TYPE_CHECKING:
    from decikit._engine.run import Engine, Executable, FusedEngine, InterpretedEngine, NumbaEngine, SteppedEngine
    from decikit.steps.scorecard import ScorecardConfig
    from decikit.steps.tables import DecisionTableConfig
    from decikit.steps.trees import TreeConfig

try:
    __version__ = _version("decikit")
except PackageNotFoundError:  # a source checkout that was never installed
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    # build
    "step", "frame_step", "flow", "dag", "branch", "loop", "each", "optimise", "EachMode", "allow_fallback",
    "helper", "param", "param_table", "missing_as", "ParamTable", "PolarsSchema", "TreeConfig", "DecisionTableConfig", "ScorecardConfig",
    # run
    "Engine", "InterpretedEngine", "SteppedEngine", "FusedEngine", "Executable", "load_flow", "describe",
    # extend
    "Step", "ConfigurableStep", "CustomFunction", "Value", "ParamRef", "StepRef", "NumbaEngine",
    # errors
    "DecikitError", "FallbackWarning", "HiddenStateWarning", "ShadowedTableWarning",
]

# Names whose modules import numba load on first use, so `import decikit` stays light.
_LAZY = {
    "Engine": "decikit._engine.run",
    "NumbaEngine": "decikit._engine.run",
    "InterpretedEngine": "decikit._engine.run",
    "SteppedEngine": "decikit._engine.run",
    "FusedEngine": "decikit._engine.run",
    "Executable": "decikit._engine.run",
    "TreeConfig": "decikit.steps.trees",
    "DecisionTableConfig": "decikit.steps.tables",
    "ScorecardConfig": "decikit.steps.scorecard",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'decikit' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
