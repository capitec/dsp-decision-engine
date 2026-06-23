"""Tests for GraphModule discriminated union registration."""

import polars as pl
import pytest

from decider.modules import GraphModule, register_graph_module
from decider.modules.functional import generate_from_functions


def _unique_name(base: str) -> str:
    import uuid
    return f"{base}_{uuid.uuid4().hex[:6]}"


def test_registered_module_validates_from_dict():
    name = _unique_name("reg_scorer")

    def score(amount: pl.Expr) -> pl.Expr:
        return amount * 2

    M = generate_from_functions(name, score)
    register_graph_module(M)

    validated = GraphModule.model_validate({"type": name, "name": "m"}).root
    assert validated.type == name


def test_re_registration_replaces_not_duplicates():
    name = _unique_name("rereg")

    def v1(amount: pl.Expr) -> pl.Expr:
        return amount * 1

    def v2(amount: pl.Expr) -> pl.Expr:
        return amount * 99

    M1 = generate_from_functions(name, v1)
    register_graph_module(M1)

    M2 = generate_from_functions(name, v2)
    register_graph_module(M2)

    # should not raise "Value X mapped to multiple choices"
    validated = GraphModule.model_validate({"type": name, "name": "m"}).root
    df = pl.DataFrame({"amount": [1.0]})
    result = validated({"input": df})
    # v2 replaced v1 — output column is named after the function (v2)
    assert result["v2"][0] == pytest.approx(99.0)


def test_multiple_distinct_modules_coexist():
    n1 = _unique_name("mod_a")
    n2 = _unique_name("mod_b")

    def fa(x: pl.Expr) -> pl.Expr:
        return x + 1

    def fb(x: pl.Expr) -> pl.Expr:
        return x + 2

    Ma = generate_from_functions(n1, fa)
    Mb = generate_from_functions(n2, fb)
    register_graph_module(Ma)
    register_graph_module(Mb)

    a = GraphModule.model_validate({"type": n1, "name": "a"}).root
    b = GraphModule.model_validate({"type": n2, "name": "b"}).root
    assert a.type == n1
    assert b.type == n2
