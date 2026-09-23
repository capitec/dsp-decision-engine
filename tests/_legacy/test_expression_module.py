"""Tests for generate_from_functions and ExpressionModule."""

import polars as pl
import pytest
from pydantic import BaseModel

from decider.modules.functional import generate_from_functions


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_df():
    return pl.DataFrame({"amount": [10.0, 20.0, 30.0], "quantity": [1, 2, 3]})


# ── basic execution ───────────────────────────────────────────────────────────

def test_single_function(simple_df):
    def doubled(amount: pl.Expr) -> pl.Expr:
        return amount * 2

    M = generate_from_functions("doubled", doubled)
    result = M(name="m")({"input": simple_df})
    assert result["doubled"].to_list() == [20.0, 40.0, 60.0]


def test_output_column_named_after_function(simple_df):
    def my_score(amount: pl.Expr) -> pl.Expr:
        return amount + 1

    M = generate_from_functions("scorer", my_score)
    result = M(name="m")({"input": simple_df})
    assert "my_score" in result.columns


def test_input_columns_preserved(simple_df):
    def doubled(amount: pl.Expr) -> pl.Expr:
        return amount * 2

    M = generate_from_functions("doubled", doubled)
    result = M(name="m")({"input": simple_df})
    assert "amount" in result.columns
    assert "quantity" in result.columns


# ── dependency wiring ─────────────────────────────────────────────────────────

def test_sibling_dependency_wired_automatically(simple_df):
    def cart_value(amount: pl.Expr, quantity: pl.Expr) -> pl.Expr:
        return amount * quantity

    def discount(cart_value: pl.Expr) -> pl.Expr:
        return pl.when(cart_value > 30).then(pl.lit(0.1)).otherwise(pl.lit(0.0))

    M = generate_from_functions("cart", cart_value, discount)
    result = M(name="m")({"input": simple_df})
    assert result["cart_value"].to_list() == [10.0, 40.0, 90.0]
    assert result["discount"][0] == pytest.approx(0.0)
    assert result["discount"][1] == pytest.approx(0.1)


def test_multi_level_dependency_chain(simple_df):
    def a(amount: pl.Expr) -> pl.Expr:
        return amount * 2

    def b(a: pl.Expr) -> pl.Expr:
        return a + 1

    def c(b: pl.Expr) -> pl.Expr:
        return b * 10

    M = generate_from_functions("chain", a, b, c)
    result = M(name="m")({"input": simple_df})
    # amount=10 → a=20 → b=21 → c=210
    assert result["c"][0] == pytest.approx(210.0)


# ── config injection ──────────────────────────────────────────────────────────

def test_config_fields_on_module():
    class Cfg(BaseModel):
        weight: float = 1.0
        offset: float = 0.0

    def score(amount: pl.Expr, config: Cfg) -> pl.Expr:
        return amount * config.weight + config.offset

    M = generate_from_functions("scorer", score)
    m = M(name="m", weight=3.0, offset=5.0)
    assert m.weight == 3.0
    assert m.offset == 5.0


def test_config_values_affect_output(simple_df):
    class Cfg(BaseModel):
        weight: float = 1.0

    def score(amount: pl.Expr, config: Cfg) -> pl.Expr:
        return amount * config.weight

    M = generate_from_functions("scorer", score)
    result = M(name="m", weight=2.0)({"input": simple_df})
    assert result["score"].to_list() == [20.0, 40.0, 60.0]


def test_config_default_used_when_not_overridden(simple_df):
    class Cfg(BaseModel):
        weight: float = 5.0

    def score(amount: pl.Expr, config: Cfg) -> pl.Expr:
        return amount * config.weight

    M = generate_from_functions("scorer", score)
    result = M(name="m")({"input": simple_df})
    assert result["score"][0] == pytest.approx(50.0)


# ── union (&) ─────────────────────────────────────────────────────────────────

def test_union_merges_output_columns(simple_df):
    def doubled(amount: pl.Expr) -> pl.Expr:
        return amount * 2

    def tripled(amount: pl.Expr) -> pl.Expr:
        return amount * 3

    M1 = generate_from_functions("m1", doubled)
    M2 = generate_from_functions("m2", tripled)
    combined = M1(name="a") & M2(name="b")
    result = combined({"input": simple_df})
    assert "doubled" in result.columns
    assert "tripled" in result.columns


# ── type discriminator ────────────────────────────────────────────────────────

def test_type_identifier_is_module_name():
    def f(x: pl.Expr) -> pl.Expr:
        return x

    M = generate_from_functions("my_module", f)
    assert M._CLASS_TYPE_IDENTIFIER == "my_module"
    assert M(name="m").type == "my_module"
