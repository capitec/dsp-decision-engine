"""Branches in every mode: bool and int conditions, silent arms, multi-step arms, nesting, lazy params."""
import polars as pl
import pytest

from decider import branch, dag, flow, frame_step, param, step
from decider.engine import Engine
from decider.exceptions import WiringError
from decider.testing import assert_equivalent

MODES = ("interpreted", "stepped", "fused")


def packed(pipeline, frame, **engine_kwargs):
    """The branches and loops a fused run of `pipeline` packed into one kernel each."""
    exe = Engine(**engine_kwargs).bind(pipeline, "fused")
    exe.run(frame)
    return sorted(exe.runner.packed)


@step(output="is_private_sector")
def sector_is_private(sector: str, private: str = param("private")) -> bool:
    return sector == private


def is_private(is_private_sector: bool) -> bool:
    return is_private_sector


@step(output="term_cap")
def cap_for_private(term_cap: float, private_cap: float = param(48.0, ge=6, le=60)) -> float:
    return min(term_cap, private_cap)


@step(output="term_cap")
def cap_for_public(term_cap: float, public_cap: float = param(60.0, ge=6, le=60)) -> float:
    return min(term_cap, public_cap)


def public_noop(is_private_sector: bool) -> bool:
    return is_private_sector


@step(output="term_cap")
def cap_for_private_as_int(term_cap: float, private_cap: float = param(48.0)) -> int:
    return int(min(term_cap, private_cap))


TERM = pl.DataFrame({"sector": ["private", "public", "private"], "term_cap": [60.0, 60.0, 30.0]})


def test_both_arms_write_the_modified_value():
    rules = branch(is_private, cap_for_private, cap_for_public, modifies=["term_cap"], name="rules")
    out = assert_equivalent(flow(sector_is_private, rules), TERM)
    assert out["term_cap"].to_list() == [48.0, 60.0, 30.0]


def test_an_arm_silent_about_a_modified_name_passes_it_through():
    rules = branch(is_private, cap_for_private, public_noop, modifies=["term_cap"], name="rules")
    out = assert_equivalent(flow(sector_is_private, rules), TERM.with_columns(term_cap=pl.Series([60.0, 55.0, 30.0])))
    assert out["term_cap"].to_list() == [48.0, 55.0, 30.0]


def test_emitting_the_condition_by_path_reports_the_arm_each_row_took():
    rules = branch(band_of, band0, band1, band2, modifies=["price"], name="rules")
    pipeline = flow(rules).emit("band_of@rules/band_of")
    out = assert_equivalent(pipeline, BANDS)
    assert out["band_of@rules/band_of"].to_list() == [0, 1, 2, 1]
    # A value emitted from inside the branch keeps it out of a packed kernel.
    assert packed(pipeline, BANDS) == []


def test_arms_disagreeing_on_a_modified_type_is_an_error():
    rules = branch(is_private, cap_for_private_as_int, cap_for_public, modifies=["term_cap"], name="rules")
    with pytest.raises(WiringError, match="disagree on the type of 'term_cap'"):
        flow(sector_is_private, rules).run(TERM)


def band_of(band: int) -> int:
    return band


@step(output="price")
def price_by_band(price: float, band: int, flag: bool, bump: float = param(0.5)) -> float:
    # A tuple indexed by `band` needs a real int, and `flag` a real bool.
    return price + (10.0, 20.0, 30.0)[band] + (bump if flag else 0.0)


@step(output="price")
def band0(price: float) -> float:
    return 1.0 + 0.0 * price


@step(output="price")
def band1(price: float) -> float:
    return 2.0 + 0.0 * price


@step(output="price")
def band2(price: float) -> float:
    return 3.0 + 0.0 * price


BANDS = pl.DataFrame({"band": [0, 1, 2, 1], "flag": [True, False, True, False], "price": [0.0, 0.0, 0.0, 100.0]})


def test_an_int_condition_picks_the_arm_by_index():
    rules = branch(band_of, band0, band1, band2, modifies=["price"], name="rules")
    assert assert_equivalent(flow(rules), BANDS)["price"].to_list() == [1.0, 2.0, 3.0, 2.0]


def test_arms_receive_their_declared_types():
    rules = branch(band_of, price_by_band, band1, band2, modifies=["price"], name="rules")
    assert assert_equivalent(flow(rules), BANDS)["price"].to_list() == [10.5, 2.0, 3.0, 2.0]


@pytest.mark.parametrize("mode", MODES)
def test_an_int_condition_out_of_range_is_an_error_naming_the_branch(mode):
    rules = branch(band_of, band0, band1, modifies=["price"], name="rules")
    with pytest.raises(ValueError, match="branch rules: the condition picked arm 2 on 1 row"):
        Engine().bind(rules, mode).run(BANDS)


def test_a_branch_needs_modifies():
    with pytest.raises(WiringError, match="needs modifies"):
        branch(is_private, cap_for_private, cap_for_public, modifies=[], name="x")


def test_a_branch_needs_a_name():
    with pytest.raises(ValueError):
        branch(is_private, cap_for_private, cap_for_public, modifies=["term_cap"], name="")


# --- multi-step arms -------------------------------------------------------------


@step(output="income_after_tax")
def after_tax(income: float, rate: float = param(0.25)) -> float:
    return income * (1.0 - rate)


@step(output="limit")
def limit_from_income(income_after_tax: float, multiple: float = param(3.0)) -> float:
    return income_after_tax * multiple


@step(output="limit")
def flat_limit(income: float) -> float:
    return 1000.0 + 0.0 * income


@step(output="band")
def band_from_limit(limit: float) -> int:
    return 2 if limit > 5000.0 else 1


@step(output="band")
def no_band(income: float) -> int:
    return 0 * int(income)


def is_employed(employed: bool) -> bool:
    return employed


EMPLOYMENT = pl.DataFrame({"income": [4000.0, 900.0, 1500.0], "employed": [True, False, True]})


@pytest.mark.parametrize("arm", [
    flow(after_tax, limit_from_income, band_from_limit, name="assessed"),
    dag(band_from_limit, limit_from_income, after_tax, name="assessed"),
], ids=["flow", "dag"])
def test_an_arm_of_several_steps_runs_them_all_on_its_rows(arm):
    fallback = flow(flat_limit, no_band, name="flat")
    rules = branch(is_employed, arm, fallback, modifies=["limit", "band"], name="rules")
    out = assert_equivalent(flow(rules), EMPLOYMENT)
    assert out["limit"].to_list() == [9000.0, 1000.0, 3375.0]
    assert out["band"].to_list() == [2, 0, 1]
    assert packed(flow(rules), EMPLOYMENT) == ["rules"]


def test_a_branch_nested_in_an_arm_packs_with_it_and_a_sibling_arm_reads_the_input():
    inner = branch(band_of, band0, band1, band2, modifies=["price"], name="inner")
    outer = branch(is_employed, inner, band0, modifies=["price"], name="outer")
    frame = BANDS.with_columns(employed=pl.Series([True, True, False, True]))
    out = assert_equivalent(flow(outer), frame)
    assert out["price"].to_list() == [1.0, 2.0, 1.0, 2.0]
    assert packed(flow(outer), frame) == ["outer", "outer/inner"]


# --- lazy params validation --------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_an_invalid_param_in_an_arm_no_row_takes_never_fails_lazily(mode):
    rules = branch(is_employed, flow(after_tax, limit_from_income, name="assessed"), flat_limit,
                   modifies=["limit"], name="rules")
    exe = Engine(params_validation="lazy").bind(flow(rules), mode)
    everyone_unemployed = EMPLOYMENT.with_columns(employed=pl.lit(False))
    bad = {"rules": {"assessed": {"after_tax": {"rate": "not a number"}}}}
    assert exe.run(everyone_unemployed, params=bad)["limit"].to_list() == [1000.0] * 3
    assert exe.report.invalid == []
    with pytest.raises(ValueError, match="rules/assessed/after_tax"):
        exe.run(EMPLOYMENT, params=bad)


def test_lazy_validation_keeps_arms_with_params_out_of_packed_kernels():
    rules = branch(is_employed, flow(after_tax, limit_from_income, name="assessed"), flat_limit,
                   modifies=["limit"], name="rules")
    assert packed(flow(rules), EMPLOYMENT) == ["rules"]
    assert packed(flow(rules), EMPLOYMENT, params_validation="lazy") == []


# --- nulls -----------------------------------------------------------------------


def test_a_null_passed_through_a_silent_arm_stays_null_in_every_mode():
    rules = branch(is_private, cap_for_private, public_noop, modifies=["term_cap"], name="rules")
    frame = TERM.with_columns(term_cap=pl.Series([60.0, None, 30.0]))
    assert assert_equivalent(flow(sector_is_private, rules), frame)["term_cap"].to_list() == [48.0, None, 30.0]


def test_a_modified_column_missing_from_the_frame_passes_a_silent_arm_as_null():
    rules = branch(is_private, cap_for_private, public_noop, modifies=["term_cap"], name="rules")
    frame = pl.DataFrame({"sector": ["public", "public"]})
    assert assert_equivalent(flow(sector_is_private, rules), frame)["term_cap"].to_list() == [None, None]


@pytest.mark.parametrize("mode", MODES)
def test_a_null_a_taken_arm_requires_is_an_error(mode):
    rules = branch(is_private, cap_for_private, cap_for_public, modifies=["term_cap"], name="rules")
    frame = TERM.with_columns(term_cap=pl.Series([60.0, None, 30.0]))
    with pytest.raises(ValueError, match="'term_cap'.*rules/cap_for_public"):
        Engine().bind(flow(sector_is_private, rules), mode).run(frame)


# --- what doesn't pack -------------------------------------------------------------


def minus_two(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(limit=pl.col("income") - 2.0)


def test_a_branch_with_a_frame_step_inside_runs_unpacked_with_the_same_answers():
    rules = branch(is_employed, frame_step(minus_two, reads=["income"], writes=["limit"]), flat_limit,
                   modifies=["limit"], name="rules")
    out = assert_equivalent(flow(rules), EMPLOYMENT)
    assert out["limit"].to_list() == [3998.0, 1000.0, 1498.0]
    assert packed(flow(rules), EMPLOYMENT) == []


def test_a_branch_with_a_nullable_output_inside_runs_unpacked():
    @step(output="limit")
    def maybe(income: float) -> float | None:
        return None if income < 2000.0 else income

    rules = branch(is_employed, maybe, flat_limit, modifies=["limit"], name="rules")
    assert assert_equivalent(flow(rules), EMPLOYMENT)["limit"].to_list() == [4000.0, 1000.0, None]
    assert packed(flow(rules), EMPLOYMENT) == []
