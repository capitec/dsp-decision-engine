"""The flagship pipeline, interpreted: answers, additive output, emit/drop, retuning and score."""
from __future__ import annotations

import polars as pl
import pytest

from decider import flow, param
from decider.engine import Engine
from decider.engine.params import ParamsError


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(
    term_cap: float,
    min_net_salary: float,
    cap: float = param(48.0, ge=6, le=60),
    income_threshold: float = param(5000.0, ge=0),
) -> float:
    if min_net_salary < income_threshold:
        return min(term_cap, cap)
    return term_cap


pipeline = flow(disposable_income, affordability_ratio, cap_by_income_band)

FRAME = pl.DataFrame({
    "net_income":     [9200.0, 4100.0, 15000.0, 4999.0],
    "expenses":       [3100.0, 1500.0,  6000.0, 2000.0],
    "instalment":     [1200.0,  800.0,  2500.0,  700.0],
    "term_cap":       [  60.0,   60.0,    60.0,   60.0],
    "min_net_salary": [9200.0, 4100.0, 15000.0, 4999.0],
})


def test_interpreted_mode_gives_the_flagship_answer():
    out = Engine().bind(pipeline, mode="interpreted").run(FRAME)
    assert out["cap_by_income_band"].to_list() == [60.0, 48.0, 60.0, 48.0]


def test_step_run_is_the_interpreted_engine():
    assert pipeline.run(FRAME).equals(Engine().bind(pipeline).run(FRAME))


def test_an_unknown_mode_is_an_error_listing_the_modes():
    with pytest.raises(ValueError, match="unknown mode 'fused'.*interpreted"):
        Engine().bind(pipeline, mode="fused")


def test_output_is_additive():
    out = pipeline.run(FRAME)
    assert out.columns[:len(FRAME.columns)] == FRAME.columns
    assert out["affordability_ratio"].to_list() == [5.083333333333333, 3.25, 3.6, 4.284285714285715]


def test_untapped_intermediates_are_not_materialised():
    assert "disposable_income" not in pipeline.run(FRAME).columns


def test_emit_materialises_an_intermediate():
    out = pipeline.emit("disposable_income").run(FRAME)
    assert out["disposable_income"].to_list() == [6100.0, 2600.0, 9000.0, 2999.0]


def test_drop_removes_an_input_column():
    assert "min_net_salary" not in pipeline.drop("min_net_salary").run(FRAME).columns


def test_drop_removes_an_output():
    assert "affordability_ratio" not in pipeline.drop("affordability_ratio").run(FRAME).columns


def test_an_unread_frame_column_passes_through_unless_dropped():
    frame = FRAME.with_columns(pl.Series("client_id", [1, 2, 3, 4]))
    assert pipeline.run(frame)["client_id"].to_list() == [1, 2, 3, 4]
    assert "client_id" not in pipeline.drop("client_id").run(frame).columns


def test_retuning_changes_the_answer_without_editing_code():
    out = pipeline.run(FRAME, params={"cap_by_income_band": {"cap": 36.0}})
    assert out["cap_by_income_band"].to_list() == [60.0, 36.0, 60.0, 36.0]


def test_a_param_outside_its_bounds_is_rejected_at_run_time_naming_node_param_and_rows():
    with pytest.raises(ParamsError, match=r"(?s)4 rows.*cap_by_income_band: param 'cap'"):
        pipeline.run(FRAME, params={"cap_by_income_band": {"cap": 999.0}})


def test_score_takes_a_dict_and_returns_a_dict():
    out = Engine().bind(pipeline).score({
        "net_income": 4100.0, "expenses": 1500.0, "instalment": 800.0,
        "term_cap": 60.0, "min_net_salary": 4100.0,
    })
    assert out["cap_by_income_band"] == 48.0


def test_score_agrees_with_run_row_for_row():
    exe = Engine().bind(pipeline)
    batch = exe.run(FRAME)["cap_by_income_band"].to_list()
    single = [exe.score(dict(zip(FRAME.columns, row)))["cap_by_income_band"] for row in FRAME.iter_rows()]
    assert batch == single
