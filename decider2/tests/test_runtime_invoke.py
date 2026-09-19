"""Scratch tests for decider2.runtime.invoke — doc 02 §3.5 (apply/score, one
kernel), doc 03 §7 (additive frame), doc 05 §2 (nulls), doc 05 §9
(acceptance criteria).

These reproduce tests/test_flagship.py's data and expectations directly
against `invoke.apply()`/`invoke.score()` and hand-built `Step`/`Interface`
objects, bypassing `flow()`/`module()`/`param()` (graph/params layer, not
mine to build) — proof that compile/+runtime/ are correct in isolation,
ahead of the full authoring surface existing.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from decider2.runtime import invoke
from decider2.types import Input, Interface, MissingInputPolicy, NullPolicy, ParamDecl, Step


def disposable_income(net_income: float, expenses: float) -> float:
    """Income remaining after committed expenses."""
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    """Affordability ratio."""
    return disposable_income / instalment


def cap_by_income_band(
    term_cap: float,
    min_net_salary: float,
    cap: float = 48.0,
    income_threshold: float = 5000.0,
) -> float:
    """Cap term at 48 months below the income floor."""
    if min_net_salary < income_threshold:
        return min(term_cap, cap)
    return term_cap


def _flagship_steps():
    return [
        Step(name="disposable_income", fn=disposable_income, params=(),
             inputs=(Input("net_income", float), Input("expenses", float))),
        Step(name="affordability_ratio", fn=affordability_ratio, params=(),
             inputs=(Input("disposable_income", float), Input("instalment", float))),
        Step(
            name="cap_by_income_band", fn=cap_by_income_band,
            inputs=(Input("term_cap", float), Input("min_net_salary", float)),
            params=(
                ParamDecl("cap", float, 48.0, _field_info(48.0, ge=6, le=60)),
                ParamDecl("income_threshold", float, 5000.0, _field_info(5000.0, ge=0)),
            ),
        ),
    ]


def _field_info(default, **kw):
    from pydantic import Field

    return Field(default, **kw)


def optional_disposable_income(net_income: "float | None", expenses: float) -> float:
    return -expenses if net_income is None else net_income - expenses


# terminals: nothing downstream reads affordability_ratio or
# cap_by_income_band (disposable_income IS read, by affordability_ratio, so
# it is excluded) -- doc 03 §7. inputs: the five leaf columns extract_frame
# needs to know to pull from the frame (doc 05 §1) -- this mirrors what
# decider2.graph.interface.raw_interface would infer from _flagship_steps().
FLAGSHIP_INTERFACE = Interface(
    inputs=(
        Input("net_income", float), Input("expenses", float), Input("instalment", float),
        Input("term_cap", float), Input("min_net_salary", float),
    ),
    outputs=("disposable_income", "affordability_ratio", "cap_by_income_band"),
    terminals=("affordability_ratio", "cap_by_income_band"),
    params_model=None,
)


FRAME = pl.DataFrame({
    "net_income":     [9200.0, 4100.0, 15000.0, 4999.0],
    "expenses":       [3100.0, 1500.0,  6000.0, 2000.0],
    "instalment":     [1200.0,  800.0,  2500.0,  700.0],
    "term_cap":       [  60.0,   60.0,    60.0,   60.0],
    "min_net_salary": [9200.0, 4100.0, 15000.0, 4999.0],
})


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_every_mode_produces_the_same_answer(tmp_path, mode):
    out = invoke.apply(_flagship_steps(), FRAME, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], mode=mode, build_dir=tmp_path)
    assert out["cap_by_income_band"].to_list() == [60.0, 48.0, 60.0, 48.0]


def test_the_three_modes_agree_exactly(tmp_path):
    frames = [
        invoke.apply(_flagship_steps(), FRAME, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], mode=m, build_dir=tmp_path)
        for m in ("interpreted", "stepped", "fused")
    ]
    assert frames[0].equals(frames[1])
    assert frames[1].equals(frames[2])


def test_output_is_additive(tmp_path):
    out = invoke.apply(_flagship_steps(), FRAME, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], build_dir=tmp_path)
    for col in FRAME.columns:
        assert col in out.columns
    assert "cap_by_income_band" in out.columns


def test_untapped_intermediate_is_not_materialised(tmp_path):
    """disposable_income feeds affordability_ratio and nothing else asks
    for it -- it must not reach the output frame (doc 03 §7), even though
    (per test_compile_driver.py) it DOES exist as an internal numpy array
    crossing the disposable_income -> affordability_ratio kernel boundary."""
    out = invoke.apply(_flagship_steps(), FRAME, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], build_dir=tmp_path)
    assert "disposable_income" not in out.columns


def test_emit_materialises_an_intermediate(tmp_path):
    out = invoke.apply(
        _flagship_steps(), FRAME, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], emit=("disposable_income",), build_dir=tmp_path
    )
    assert out["disposable_income"].to_list() == [6100.0, 2600.0, 9000.0, 2999.0]


def test_drop_removes_a_declared_column(tmp_path):
    out = invoke.apply(
        _flagship_steps(), FRAME, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], drop=("min_net_salary",), build_dir=tmp_path
    )
    assert "min_net_salary" not in out.columns


def test_retuning_changes_the_answer_without_editing_code(tmp_path):
    out = invoke.apply(
        _flagship_steps(), FRAME, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2],
        params={"cap_by_income_band": {"cap": 36.0}}, build_dir=tmp_path,
    )
    assert out["cap_by_income_band"].to_list() == [60.0, 36.0, 60.0, 36.0]


def test_a_param_outside_its_bounds_is_rejected_by_pydantic(tmp_path):
    with pytest.raises(Exception):
        invoke.apply(
            _flagship_steps(), FRAME, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2],
            params={"cap_by_income_band": {"cap": 999.0}}, build_dir=tmp_path,
        )


def test_score_takes_a_dict_and_returns_a_dict(tmp_path):
    out = invoke.score(
        _flagship_steps(),
        {"net_income": 4100.0, "expenses": 1500.0, "instalment": 800.0,
         "term_cap": 60.0, "min_net_salary": 4100.0},
        interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], build_dir=tmp_path,
    )
    assert out["cap_by_income_band"] == 48.0


def test_score_agrees_with_apply_row_for_row(tmp_path):
    batch = invoke.apply(_flagship_steps(), FRAME, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], build_dir=tmp_path)
    batch_values = batch["cap_by_income_band"].to_list()
    single = [
        invoke.score(_flagship_steps(), dict(zip(FRAME.columns, row)), interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], build_dir=tmp_path)[
            "cap_by_income_band"
        ]
        for row in FRAME.iter_rows()
    ]
    assert batch_values == single


def test_score_and_apply_use_the_same_compiled_kernel(tmp_path):
    """doc 05 §9 acceptance criterion 2, literally: build once, use the
    returned driver's kernel for both a batch and a single record."""
    from decider2.compile.driver import build_driver

    steps = _flagship_steps()
    driver = build_driver(steps, [0, 1, 2], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    cap_kernel = driver.segments[2].kernel_fn

    out = invoke.apply(steps, FRAME, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], build_dir=tmp_path)
    record = invoke.score(
        steps,
        {"net_income": 4100.0, "expenses": 1500.0, "instalment": 800.0,
         "term_cap": 60.0, "min_net_salary": 4100.0},
        interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], build_dir=tmp_path,
    )
    # Rebuilding against the same content-addressed build_dir must reuse the
    # identical compiled dispatcher, not a fresh one (doc 05 §4.2).
    from decider2.compile.driver import build_driver as build_driver2

    driver2 = build_driver2(steps, [0, 1, 2], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    assert driver2.segments[2].kernel_fn is cap_kernel
    assert out["cap_by_income_band"].to_list()[1] == record["cap_by_income_band"]


# --- doc 05 §2 (nulls) ------------------------------------------------------


def test_required_null_raises_naming_the_column_count_and_example_rows(tmp_path):
    """decider2.boundary.nulls routes a REQUIRED null to a Decision by
    default (doc 03 §1 overriding doc 05 §2's literal hard-fail-by-default
    reading -- see decider2/boundary/nulls.py's own docstring and this
    module's report); the hard-fail message survives for a `raise_for`
    column, which is what this test actually exercises."""
    frame = FRAME.with_columns(pl.Series("net_income", [9200.0, None, 15000.0, None]))
    policy = MissingInputPolicy(raise_for=("net_income",))
    with pytest.raises(ValueError) as exc_info:
        invoke.apply(
            _flagship_steps(), frame, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2],
            policy=policy, build_dir=tmp_path,
        )
    msg = str(exc_info.value)
    assert "net_income" in msg
    assert "2 null" in msg
    assert "required" in msg


def test_required_null_routes_the_row_instead_of_raising_by_default(tmp_path):
    """The doc 03 §1 default: a REQUIRED null (not in raise_for) routes its
    row away rather than failing the whole batch -- the row still comes
    back in the output frame (never silently dropped).

    Its terminal column is NaN, not a genuine polars null: `decider2.
    boundary.writeback.DtypeGroup` carries a plain numpy array with no
    validity mask at all, so there is no way to write back an actual null
    through it (a real gap for a non-float terminal -- int64/bool have no
    NaN equivalent -- flagged in this module's report)."""
    import math

    frame = FRAME.with_columns(pl.Series("net_income", [9200.0, None, 15000.0, 4999.0]))
    out = invoke.apply(_flagship_steps(), frame, interface=FLAGSHIP_INTERFACE, group_ids=[0, 1, 2], build_dir=tmp_path)
    assert out.height == 4  # the routed row is still present
    values = out["cap_by_income_band"].to_list()
    assert math.isnan(values[1])
    assert values[0] == 60.0 and values[2] == 60.0 and values[3] == 48.0


def _interface_with_net_income(inp: Input) -> Interface:
    others = tuple(i for i in FLAGSHIP_INTERFACE.inputs if i.name != "net_income")
    return Interface(
        inputs=(inp,) + others, outputs=FLAGSHIP_INTERFACE.outputs,
        terminals=FLAGSHIP_INTERFACE.terminals, params_model=None,
    )


def test_missing_as_fill_is_applied_before_the_step_sees_it(tmp_path):
    steps = _flagship_steps()
    # net_income declares a fill of 0.0 for missing values (tier 2, doc 05 §2)
    net_income_input = Input("net_income", float, null_policy=NullPolicy.MISSING_AS, fill=0.0)
    steps[0] = Step(
        name="disposable_income", fn=disposable_income, params=(),
        inputs=(net_income_input, Input("expenses", float)),
    )
    interface = _interface_with_net_income(net_income_input)
    frame = FRAME.with_columns(pl.Series("net_income", [9200.0, None, 15000.0, 4999.0]))
    out = invoke.apply(steps, frame, interface=interface, group_ids=[0, 1, 2], emit=("disposable_income",), build_dir=tmp_path)
    assert out["disposable_income"].to_list()[1] == 0.0 - 1500.0


def test_boundary_shim_handles_a_nullable_float64_column_round_trip(tmp_path):
    """doc 05 §9 acceptance criterion 4 (partial -- Float64 only; Int64/
    Boolean round-tripping is decider2.compile.boundary's job, see this
    module's docstring)."""
    steps = _flagship_steps()
    net_income_input = Input("net_income", float, null_policy=NullPolicy.OPTIONAL)
    steps[0] = Step(
        name="disposable_income", fn=optional_disposable_income, params=(),
        inputs=(net_income_input, Input("expenses", float)),
    )
    interface = _interface_with_net_income(net_income_input)
    frame = FRAME.with_columns(pl.Series("net_income", [9200.0, None, 15000.0, 4999.0]))
    out = invoke.apply(steps, frame, interface=interface, group_ids=[0, 1, 2], emit=("disposable_income",), build_dir=tmp_path)
    assert out["disposable_income"].to_list() == [6100.0, -1500.0, 9000.0, 2999.0]
