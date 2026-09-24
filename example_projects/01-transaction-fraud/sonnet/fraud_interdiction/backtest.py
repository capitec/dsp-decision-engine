"""Batch backtest mode (spec 01 §5.17): the same pipeline, run over stored events, in bulk.

**The equivalence requirement is structural, not proven separately.** `pipeline.build()`
returns one `decider` `dag`; that one object serves both `Engine.score(record, params)`
(the real-time path) and, in principle, `Engine.run(DataFrame, params)` (batch) -- there
is no second rule-evaluation implementation to keep in sync. `tests/test_pipeline.py`
still asserts a batch and a per-record run agree over a sample, because "the same object"
is a design property, not a runtime guarantee, and §5.17's own acceptance criterion
(09 §10.6) is exactly this comparison, run continuously.

**`.run()` on a `DataFrame` is not used here -- confirmed broken for this pipeline's own
output shape, not routed around speculatively.** `fired_rule_ids`,
`fired_on_overlay_ids`, `counterfactual_fired_rule_ids` and `shadow_fired_rule_ids` are
`list[str]`, genuinely empty on most events (most rules don't fire on most events -- see
§5.18 "of 521 live rules, 60-90 have not fired in 90 days"). `Engine.run()` crashes
`decider`'s own result-materialisation path the moment a batch mixes an empty and a
non-empty `list[str]` row -- confirmed with an 8-line, pipeline-free reproduction:

    @frame_step(reads=["x"], writes=["y"])
    def f(df):
        return df.with_columns(pl.Series("y", [["a","b"], [], ["c"]], dtype=pl.List(pl.Utf8)))
    Engine().bind(f).run(pl.DataFrame({"x": [1,2,3]}), {})
    # pyo3_runtime.PanicException: ... SchemaMismatch("invalid series dtype: expected
    # `String`, got `object` for series with name ``")

The identical shape with `list[int]`, or with no empty row, both work -- see NOTES.md
"Framework friction" for the isolated repro and the two confirming variants. Since this
project cannot edit `decider` internals, batch mode here is `Engine.score()` called once
per event through one bound `Executable` (still the identical rule-evaluation path;
nothing about "the same code" is lost, only the vectorised `DataFrame` entry point),
with the per-event dicts assembled into a `DataFrame` afterwards in plain Python, which
does not go through the buggy path.

**With and without the overlay stack**: both runs go through `adjustment_stack_enabled`
(the same `param()` flip 00's `core.adjustments` document uses for its own "stack off"
run), not a second code path.
"""
from __future__ import annotations

import polars as pl

from credit_core.rounding import round_instalment

from fraud_interdiction import vocab


def run_backtest(pipeline_step, params: dict, events: list[dict], *, adjustment_stack_enabled: bool = True,
                  dag_name: str = "fraud_interdiction", stack_step_name: str = "_stack_enabled") -> pl.DataFrame:
    """Score every event through one bound pipeline, with the overlay stack on or off
    (§5.17 "with and without"). See this module's docstring for why `.score()` per event,
    not `Engine.run()` on the whole `DataFrame`.
    """
    from decider import Engine

    run_params = _with_stack_toggle(params, dag_name, stack_step_name, adjustment_stack_enabled)
    executable = Engine().bind(pipeline_step)
    rows = [executable.score(event, run_params) for event in events]
    return pl.DataFrame(rows)


def _with_stack_toggle(params: dict, dag_name: str, stack_step_name: str, enabled: bool) -> dict:
    merged = {k: dict(v) for k, v in params.items()}
    merged.setdefault(dag_name, dict(merged.get(dag_name, {})))
    merged[dag_name][stack_step_name] = {"adjustment_stack_enabled": enabled}
    return merged


def summarise(result: pl.DataFrame, outcomes: dict[int, bool] | None = None) -> dict:
    """A reduced slice of §5.17's metrics table: hit rate and rand value blocked, rounded to the
    cent (`core.rounding`) -- the two numbers a backtest committee reads first. Precision,
    incremental catch and overlap need a labelled outcome per event (§5.16, out of scope here).
    """
    n = result.height
    blocking = {vocab.ACTION_DECLINE, vocab.ACTION_HOLD_FOR_REVIEW, vocab.ACTION_BLOCK_CHANNEL,
                vocab.ACTION_FREEZE_ACCOUNT}
    blocked = result.filter(pl.col("action_code").is_in(blocking))
    hit_rate = round(blocked.height / n, 4) if n else 0.0
    value_blocked = round_instalment(float(blocked["amount"].sum())) if "amount" in result.columns else 0.0
    return {"events": n, "hit_rate": hit_rate, "value_blocked": value_blocked}
