"""Shapes beyond the flagship, interpreted: waterfalls, version chains, dtypes, strings, shadowing."""
from __future__ import annotations

import polars as pl
import pytest

from decider import flow, param, step
from decider.engine import Engine
from decider.engine.params import ParamsError


def inv(x: float) -> float:
    return 1.0 / (x - 1.0)


def test_running_never_calls_a_step_on_values_the_caller_did_not_supply(bind):
    assert bind(flow(inv)).run(pl.DataFrame({"x": [5.0, 3.0]}))["inv"].to_list() == [0.25, 0.5]


@step(output="term_cap")
def product_ceiling(requested_term: float, ceiling: float = param(60.0, ge=6, le=84)) -> float:
    return min(requested_term, ceiling)


@step(output="term_cap")
def cap_by_income(term_cap: float, min_net_salary: float,
                  cap: float = param(48.0, ge=6, le=60),
                  floor: float = param(5000.0, ge=0)) -> float:
    return min(term_cap, cap) if min_net_salary < floor else term_cap


WATERFALL_FRAME = pl.DataFrame({"requested_term": [72.0, 72.0], "min_net_salary": [9200.0, 4100.0]})


def test_a_step_may_narrow_a_value_it_also_reads(bind):
    out = bind(flow(product_ceiling, cap_by_income)).run(WATERFALL_FRAME)
    assert out["term_cap"].to_list() == [60.0, 48.0]


def test_the_version_chain_matches_the_answer(bind):
    out = bind(flow(product_ceiling, cap_by_income).emit("term_cap@*")).run(WATERFALL_FRAME)
    assert out["term_cap@product_ceiling"].to_list() == [60.0, 60.0]
    assert out["term_cap@cap_by_income"].to_list() == [60.0, 48.0]
    assert out["term_cap"].to_list() == [60.0, 48.0]


def test_a_single_version_can_be_emitted_by_its_producer(bind):
    out = bind(flow(product_ceiling, cap_by_income).emit("term_cap@product_ceiling")).run(WATERFALL_FRAME)
    assert out["term_cap@product_ceiling"].to_list() == [60.0, 60.0]


def test_a_self_read_waterfall_seeded_from_the_frame_overwrites_the_input_column(bind):
    out = bind(flow(cap_by_income)).run(pl.DataFrame({"term_cap": [60.0, 60.0], "min_net_salary": [9200.0, 4100.0]}))
    assert out.columns.count("term_cap") == 1
    assert out["term_cap"].to_list() == [60.0, 48.0]


def term_cap_a(term_cap: float) -> float:
    return term_cap - 1.0


def test_a_name_resembling_its_own_output_is_not_an_error(bind):
    assert bind(flow(term_cap_a)).run(pl.DataFrame({"term_cap": [60.0]}))["term_cap_a"].to_list() == [59.0]


def test_an_int_step_returns_an_integer_column(bind):
    def cents(rands: int) -> int:
        return rands * 100

    out = bind(flow(cents)).run(pl.DataFrame({"rands": pl.Series([12, 34], dtype=pl.Int64)}))
    assert out["cents"].dtype == pl.Int64
    assert out["cents"].to_list() == [1200, 3400]


def test_int64_above_2_to_the_53_is_not_degraded(bind):
    big = 2 ** 53 + 1

    def passthrough(n: int) -> int:
        return n + 0

    out = bind(flow(passthrough)).run(pl.DataFrame({"n": pl.Series([big], dtype=pl.Int64)}))
    assert out["passthrough"].to_list() == [big]


def test_a_bool_step_returns_a_boolean_column(bind):
    def is_thin(score: float) -> bool:
        return score < 600.0

    out = bind(flow(is_thin)).run(pl.DataFrame({"score": [550.0, 700.0]}))
    assert out["is_thin"].dtype == pl.Boolean
    assert out["is_thin"].to_list() == [True, False]


def is_private(sector: str) -> float:
    return 1.0 if sector == "private" else 0.0


def test_a_string_input_reaches_an_interpreted_step_as_a_string():
    out = flow(is_private).run(pl.DataFrame({"sector": ["private", "public", "private"]}))
    assert out["is_private"].to_list() == [1.0, 0.0, 1.0]


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_a_string_compared_with_a_literal_in_the_body_is_a_loud_error_when_compiled(mode):
    # A compiled step sees a code, which a body literal never equals: fail, never answer wrongly.
    with pytest.raises(ValueError, match="is_private: `str` input 'sector'.*declare the literal as a `str` param"):
        Engine().bind(flow(is_private), mode=mode).run(pl.DataFrame({"sector": ["private"]}))


def test_a_misspelled_param_is_rejected(bind):
    def capper(term_cap: float, cap: float = param(48.0, ge=6, le=60)) -> float:
        return min(term_cap, cap)

    with pytest.raises(ParamsError, match="unknown param 'capp'; did you mean 'cap'"):
        bind(flow(capper)).run(pl.DataFrame({"term_cap": [60.0]}), params={"capper": {"capp": 10.0}})


def test_a_frame_column_shadowing_a_step_output_is_an_error(bind):
    def disposable_income(net_income: float, expenses: float) -> float:
        return net_income - expenses

    frame = pl.DataFrame({"net_income": [9200.0], "expenses": [3100.0], "disposable_income": [-999.0]})
    with pytest.raises(ValueError, match="'disposable_income' is produced by this pipeline"):
        bind(flow(disposable_income)).run(frame)


def sector_rate(sector: str, private: str = param("private"), rate: float = param(0.9, gt=0)) -> float:
    return rate if sector == private else 1.0


SECTOR_FRAME = pl.DataFrame({"sector": ["private", "public", "private", "government"]})


def test_a_string_literal_declared_as_a_param_compares_correctly(bind):
    assert bind(flow(sector_rate)).run(SECTOR_FRAME)["sector_rate"].to_list() == [0.9, 1.0, 0.9, 1.0]


def test_changing_a_string_literal_is_a_value_change(bind):
    out = bind(flow(sector_rate)).run(SECTOR_FRAME, params={"sector_rate": {"private": "government"}})
    assert out["sector_rate"].to_list() == [1.0, 1.0, 1.0, 0.9]


def test_a_literal_absent_from_the_data_never_matches(bind):
    out = bind(flow(sector_rate)).run(SECTOR_FRAME, params={"sector_rate": {"private": "martian"}})
    assert out["sector_rate"].to_list() == [1.0, 1.0, 1.0, 1.0]
