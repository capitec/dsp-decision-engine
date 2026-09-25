"""decider: decision pipelines from plain Python functions, run over polars frames or single records.

Start here: `decider guide` prints the getting-started guide (`decider/GUIDE.md`
in the installed package). Don't read the repo's `docs/` folder: it describes
an older version. The guide and the docstrings are current.

Quickstart::

    import polars as pl
    from decider import Engine, flow, param

    def ratio(income: float, debt: float) -> float:                  # plain arguments: request data
        return debt / income

    def affordable(ratio: float, limit: float = param(0.4)) -> bool:  # param(): policy, from params.json
        return ratio <= limit

    pipeline = flow(ratio, affordable, name="afford")
    df = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})
    pipeline.run(df)                                          # income, debt, affordable
    Engine().bind(pipeline, mode="fused").score({"income": 1000.0, "debt": 200.0})   # one record, as a dict

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
`decider serve` build and serve it. Elsewhere: `decider.serving`
(`RequestHandler`), `decider.config` (`JsonFileStore`), `decider.steps.trees`,
`.tables` and `.scorecard` (`TreeConfig`, `DecisionTableConfig`,
`ScorecardConfig`), `decider.testing` (`assert_equivalent`, `no_recompile`).
Build, wiring, params and missing-input errors are `decider.exceptions.DeciderError`s.
"""
from decider.engine import Engine
from decider.engine.params import Table, missing_as, param, param_table
from decider.fields import Duration, FieldMetadata, Money, Percent
from decider.steps import ConfigurableStep, ParamRef, Value, branch, dag, flow, frame_step, loop, step

__all__ = [
    "ConfigurableStep", "Duration", "Engine", "FieldMetadata", "Money", "ParamRef", "Percent", "Value", "branch",
    "dag", "flow", "frame_step", "loop", "Table", "missing_as", "param", "param_table", "step",
]
