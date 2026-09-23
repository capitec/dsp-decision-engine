"""decider: decision pipelines from plain Python functions, run over polars frames or single records.

A function's arguments are the columns it reads and its name is the column it
writes. `flow` runs steps in written order (a later write wins), `dag` in
dependency order; `branch`, `loop` and `ConfigurableStep` subclasses (trees,
tables, rules loaded from JSON) compose the same way. Tunable values are
`param()`s, retuned per run through a params document without rebuilding.

Example::

    import polars as pl
    from decider import Engine, flow, param

    def ratio(income: float, debt: float) -> float:
        return debt / income

    def affordable(ratio: float, limit: float = param(0.4)) -> bool:
        return ratio <= limit

    pipeline = flow(ratio, affordable, name="afford")
    df = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})
    pipeline.run(df)                                          # income, debt, affordable
    pipeline.emit("ratio").run(df)                            # intermediates are dropped unless emitted
    pipeline.run(df, params={"afford": {"affordable": {"limit": 0.9}}})
    pipeline.parameters().defaults()                          # the params document to edit
    Engine().bind(pipeline).score({"income": 1000.0, "debt": 200.0})   # one record, as a dict
    s = pipeline.session(df); s.break_at("afford/affordable"); s.resume()   # debug step by step

Build, wiring, params and missing-input errors are `decider.exceptions.DeciderError`s.
"""
from decider.engine import Engine
from decider.engine.params import missing_as, param
from decider.steps import ConfigurableStep, ParamRef, Value, branch, dag, flow, frame_step, loop, step

__all__ = [
    "ConfigurableStep", "Engine", "ParamRef", "Value", "branch", "dag", "flow", "frame_step", "loop",
    "missing_as", "param", "step",
]
