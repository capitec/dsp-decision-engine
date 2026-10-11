"""`expr_condition`: a boolean expression as a one-output tree step."""
import warnings

import polars as pl
import pytest

from decikit import FallbackWarning, FusedEngine, branch, flow, step
from decikit.exceptions import ExprError
from decikit.steps.trees import expr_condition
from decikit.lab.testing import assert_equivalent


def test_numeric_comparison_against_a_threshold(run):
    out = run(expr_condition("score >= 600", name="good"), pl.DataFrame({"score": [500.0, 600.0, 700.0]}))
    assert out["good"].to_list() == [False, True, True]


def test_flipped_numeric_comparison(run):
    out = run(expr_condition("600 <= score", name="good"), pl.DataFrame({"score": [500.0, 600.0, 700.0]}))
    assert out["good"].to_list() == [False, True, True]


def test_negative_number_literal_is_a_plain_threshold(run):
    out = run(expr_condition("score >= -5", name="good"), pl.DataFrame({"score": [-10.0, -1.0]}))
    assert out["good"].to_list() == [False, True]


def test_computed_feature_from_an_arithmetic_sub_expression(run):
    df = pl.DataFrame({"income": [5000.0, 500.0], "expenses": [1000.0, 600.0]})
    out = run(expr_condition("income - expenses > 1000", name="affordable"), df)
    assert out["affordable"].to_list() == [True, False]


def test_two_columns_compared_with_no_constant_side(run):
    df = pl.DataFrame({"a": [5.0, 1.0, 3.0], "b": [3.0, 2.0, 3.0]})
    out = run(expr_condition("a > b", name="a_bigger"), df)
    assert out["a_bigger"].to_list() == [True, False, False]


def test_string_equality(run):
    df = pl.DataFrame({"crc_model": ["CrC3.0", "CrB", None]})
    out = run(expr_condition('crc_model == "CrC3.0"', name="is_crc"), df)
    assert out["is_crc"].to_list() == [True, False, False]


def test_string_inequality(run):
    df = pl.DataFrame({"crc_model": ["CrC3.0", "CrB", None]})
    out = run(expr_condition('crc_model != "CrC3.0"', name="not_crc"), df)
    assert out["not_crc"].to_list() == [False, True, False]


def test_and_or_not_and_bare_bool_name(run):
    df = pl.DataFrame({"score": [500.0, 500.0, 700.0], "flag": [False, True, False]})
    assert run(expr_condition("flag", name="o"), df)["o"].to_list() == [False, True, False]
    assert run(expr_condition("not flag", name="o"), df)["o"].to_list() == [True, False, True]
    assert run(expr_condition("score >= 600 and flag", name="o"), df)["o"].to_list() == [False, False, False]
    assert run(expr_condition("score >= 600 or flag", name="o"), df)["o"].to_list() == [False, True, True]


def test_null_numeric_input_is_false(run):
    out = run(expr_condition("score >= 600", name="good"), pl.DataFrame({"score": [None]}, schema={"score": pl.Float64}))
    assert out["good"].to_list() == [False]


def test_null_bool_input_is_false(run):
    out = run(expr_condition("flag", name="good"), pl.DataFrame({"flag": [None]}, schema={"flag": pl.Boolean}))
    assert out["good"].to_list() == [False]


def test_null_string_input_is_false(run):
    out = run(expr_condition('crc_model == "CrC3.0"', name="good"), pl.DataFrame({"crc_model": [None]}))
    assert out["good"].to_list() == [False]


def test_null_input_is_false_under_negation_via_not_equal(run):
    out = run(expr_condition('crc_model != "CrC3.0"', name="good"), pl.DataFrame({"crc_model": [None]}))
    assert out["good"].to_list() == [False]


def test_null_input_is_false_under_explicit_not(run):
    out = run(expr_condition('not (crc_model == "CrC3.0")', name="good"), pl.DataFrame({"crc_model": [None]}))
    assert out["good"].to_list() == [False]


def test_null_numeric_input_is_false_under_negation_too(run):
    out = run(expr_condition("not (score >= 600)", name="good"), pl.DataFrame({"score": [None]}, schema={"score": pl.Float64}))
    assert out["good"].to_list() == [False]


def test_negated_and_with_a_null_side_stays_uniformly_null_safe(run):
    # "not (crc_model == 'a' and score > 5)" with crc_model null: the inner "and" is false
    # regardless of the null whenever score <= 5 (score > 5 alone already fails it), but when
    # score > 5 the null decides it - and must still resolve so the whole expression is false.
    df = pl.DataFrame({"crc_model": [None, None], "score": [1.0, 10.0]})
    out = run(expr_condition('not (crc_model == "a" and score > 5)', name="good"), df)
    assert out["good"].to_list() == [True, False]


@step(output="cap")
def _cap_a(cap_a: float) -> float:
    return cap_a


@step(output="cap")
def _cap_b(cap_b: float) -> float:
    return cap_b


ARMS = pl.DataFrame({"score": [500.0, 700.0], "crc_model": ["CrA", "CrC3.0"], "cap_a": [1.0, 1.0], "cap_b": [2.0, 2.0]})


def test_used_as_a_numeric_branch_condition_agrees_across_modes():
    # score=[500, 700]: row 0's condition is false (arm1, cap_b), row 1's is true (arm0, cap_a).
    rules = branch(expr_condition("score >= 600", name="good"), _cap_a, _cap_b, modifies=["cap"], name="b")
    out = assert_equivalent(flow(rules, name="p"), ARMS)
    assert out["cap"].to_list() == [2.0, 1.0]


def test_used_as_a_string_branch_condition_agrees_across_modes():
    rules = branch(expr_condition('crc_model == "CrC3.0"', name="good"), _cap_a, _cap_b, modifies=["cap"], name="b")
    out = assert_equivalent(flow(rules, name="p"), ARMS)
    assert out["cap"].to_list() == [2.0, 1.0]


def test_numeric_condition_compiles_without_falling_back():
    rules = branch(expr_condition("score >= 600", name="good"), _cap_a, _cap_b, modifies=["cap"], name="b")
    exe = FusedEngine(strict_compile=True).bind(flow(rules, name="p"))
    exe.run(ARMS)
    assert exe.fallbacks() == {}


def test_string_condition_compiles_without_falling_back():
    # Trees read strings as byte spans inside their kernel, so an exact string_match
    # compiles too: no FallbackWarning and no fallback under strict_compile.
    rules = branch(expr_condition('crc_model == "CrC3.0"', name="good"), _cap_a, _cap_b, modifies=["cap"], name="b")
    exe = FusedEngine(strict_compile=True).bind(flow(rules, name="p"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exe.run(ARMS)
    assert not any(issubclass(w.category, FallbackWarning) for w in caught)
    assert exe.fallbacks() == {}


@pytest.mark.parametrize("bad, match", [
    ("score + 'x' > 5", "str literal"),
    ("foo(score) > 5", "not admitted"),
    ("score.value > 5", "attribute access"),
    ("", "empty"),
])
def test_rejected_expressions_raise_expr_error(bad, match):
    with pytest.raises(ExprError, match=match):
        expr_condition(bad, name="bad")
