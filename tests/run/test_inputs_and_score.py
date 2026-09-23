"""Input dtypes survive score(), nulls follow each input's policy, and step errors propagate."""
from __future__ import annotations

import polars as pl
import pytest

from decider import flow, missing_as
from decider.engine import Engine
from decider.engine.boundary.nulls import MissingInputError


def test_score_preserves_int_precision_above_2_53():
    def big(n: int) -> int:
        return n * 3 + 1

    exe = Engine().bind(flow(big))
    n = 2 ** 53 + 1
    assert exe.score({"n": n})["big"] == exe.run(pl.DataFrame({"n": [n]}))["big"][0] == n * 3 + 1


def test_score_types_a_bool_input_as_bool():
    def flag(active: bool) -> bool:
        return not active

    assert Engine().bind(flow(flag)).score({"active": True})["flag"] is False


def test_score_can_index_a_tuple_by_an_int_input():
    def lookup(term: int) -> float:
        return (0.05, 0.06, 0.07, 0.08)[term]

    exe = Engine().bind(flow(lookup))
    via_run = exe.run(pl.DataFrame({"term": [0, 1, 2, 3]}))["lookup"].to_list()
    assert [exe.score({"term": t})["lookup"] for t in range(4)] == via_run == [0.05, 0.06, 0.07, 0.08]


def test_an_int_input_declared_float_is_cast():
    def half(x: float) -> float:
        return x / 2

    assert flow(half).run(pl.DataFrame({"x": [3, 4]}))["half"].to_list() == [1.5, 2.0]


def test_a_step_error_propagates():
    def always_broken(x: float) -> float:
        raise RuntimeError("deliberately broken")

    with pytest.raises(RuntimeError, match="deliberately broken"):
        flow(always_broken).run(pl.DataFrame({"x": [1.0, 2.0]}))


def affordability(instalment: float) -> float:
    return instalment * 0.35


def filled(bureau_score: float = missing_as(0.0)) -> float:
    return bureau_score * 0.01


def test_score_raises_naming_a_required_input_absent_from_the_record():
    with pytest.raises(MissingInputError, match="input 'instalment' of step 'affordability' .* not in the input frame"):
        Engine().bind(flow(affordability)).score({})


def test_score_fills_a_missing_as_input_absent_from_the_record():
    assert Engine().bind(flow(filled)).score({})["filled"] == 0.0


def test_run_raises_naming_a_required_column_absent_from_the_frame():
    with pytest.raises(MissingInputError, match="'instalment'.*not in the input frame"):
        flow(affordability).run(pl.DataFrame({"id": [1, 2, 3]}))


def test_run_fills_a_missing_as_column_absent_from_the_frame():
    assert flow(filled).run(pl.DataFrame({"id": [1, 2, 3]}))["filled"].to_list() == [0.0, 0.0, 0.0]


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def filled_income(expenses: float, net_income: float = missing_as(0.0)) -> float:
    return net_income - expenses


def optional_income(net_income: float | None, expenses: float) -> float:
    return -expenses if net_income is None else net_income - expenses


FRAME = pl.DataFrame({
    "net_income": [9200.0, None, 15000.0, None],
    "expenses":   [3100.0, 1500.0, 6000.0, 2000.0],
})


def test_a_required_null_raises_naming_the_input_the_step_and_the_null_count():
    with pytest.raises(MissingInputError, match="input 'net_income' of step 'disposable_income'.*2 null row"):
        flow(disposable_income).run(FRAME)


def test_a_missing_as_null_is_filled_before_the_step_sees_it():
    assert flow(filled_income).run(FRAME)["filled_income"].to_list() == [6100.0, -1500.0, 9000.0, -2000.0]


def test_an_optional_null_reaches_the_step_as_none():
    assert flow(optional_income).run(FRAME)["optional_income"].to_list() == [6100.0, -1500.0, 9000.0, -2000.0]


def test_each_reader_applies_its_own_null_policy_to_one_column():
    def strict(net_income: float) -> float:
        return net_income

    with pytest.raises(MissingInputError, match="step 'strict'"):
        flow(optional_income, strict).run(FRAME)


def test_an_optional_input_keeps_its_nulls_in_the_output():
    def keep(x: float | None) -> float | None:
        return x

    assert flow(keep).run(pl.DataFrame({"x": [1.0, None]}))["keep"].to_list() == [1.0, None]
