"""`CustomFunction`: a step whose fields are data and whose logic is a staticmethod."""
from __future__ import annotations

import typing as t
import warnings

import polars as pl
import pytest
from pydantic import TypeAdapter

from decider import ConfigurableStep, CustomFunction, FallbackWarning, Value, allow_fallback, flow, missing_as, step
from decider.exceptions import MissingInputError
from decider.lab import diff
from decider.lab.testing import assert_equivalent
from decider.steps import StepRef


class Cap(CustomFunction):
    type: t.Literal["custom_cap"] = "custom_cap"
    x: str
    limit: Value[float] = 100.0

    @staticmethod
    def impl(x: float, limit: float) -> float:
        return min(x, limit)


class Spread(CustomFunction):
    type: t.Literal["custom_spread"] = "custom_spread"
    x: str
    centre: Value[float]

    @staticmethod
    def impl(x: float, centre: float) -> tuple[float, float]:
        return x - centre, abs(x - centre)


FRAME = pl.DataFrame({"amount": [50.0, 150.0, 250.0]})


def _pipeline(**fields):
    return flow(Cap(name="capped", x="amount", **fields), name="p")


def test_a_custom_function_compiles_and_every_mode_gives_the_same_answer():
    out = assert_equivalent(_pipeline(limit=100.0), FRAME)
    assert out["capped"].to_list() == [50.0, 100.0, 100.0]
    assert _pipeline().compile("fused").fallbacks() == {}


def test_a_column_field_reads_the_column_it_names():
    other = pl.DataFrame({"balance": [500.0]})
    assert flow(Cap(name="c", x="balance", limit=70.0)).run(other)["c"].to_list() == [70.0]


def test_a_value_field_is_a_param_when_given_a_ref():
    pipe = _pipeline(limit={"param": "limit", "default": 100.0})
    assert pipe.parameters()["p/capped"]["limit"]["default"] == 100.0
    assert assert_equivalent(pipe, FRAME, params={"p": {"capped": {"limit": 60.0}}})["capped"].to_list() == [50.0, 60.0, 60.0]


def test_the_same_data_twice_diffs_identical_and_a_changed_value_does_not():
    assert diff.structure(_pipeline(limit=100.0), _pipeline(limit=100.0)).identical
    assert not diff.structure(_pipeline(limit=100.0), _pipeline(limit=200.0)).identical


def _versioned(one: bool):
    # Two bodies under one qualified name: the types are equal, only the code differs.
    if one:
        class Versioned(CustomFunction):
            x: str

            @staticmethod
            def impl(x: float) -> float:
                return x + 1.0
    else:
        class Versioned(CustomFunction):
            x: str

            @staticmethod
            def impl(x: float) -> float:
                return x + 2.0

    return flow(Versioned(name="v", x="amount"), name="p")


def test_a_changed_impl_body_diffs_different():
    assert diff.structure(_versioned(True), _versioned(True)).identical
    assert not diff.structure(_versioned(True), _versioned(False)).identical


def test_it_loads_from_a_config_document_and_dumps_the_same_document():
    doc = {"type": "custom_cap", "name": "capped", "x": "amount", "limit": {"param": "k", "default": 50.0}}
    loaded = ConfigurableStep.load(doc)
    assert type(loaded) is Cap and loaded.limit.default == 50.0
    assert ConfigurableStep.load(loaded.model_dump_json()) == loaded
    assert loaded.model_dump(exclude_defaults=True)["limit"] == {"param": "k", "default": 50.0}
    assert flow(loaded).run(FRAME)["capped"].to_list() == [50.0, 50.0, 50.0]


def test_a_custom_function_nested_in_a_field_keeps_its_own_fields_on_dump():
    cap = Cap(name="capped", x="amount", limit=5.0)
    again = TypeAdapter(StepRef).validate_json(cap.model_dump_json())
    assert again == cap


def test_several_outputs_come_from_a_tuple_return_and_the_outputs_field():
    spread = Spread(name="s", x="amount", centre=100.0, outputs=["gap", "dist"])
    out = assert_equivalent(flow(spread, name="p"), FRAME)
    assert out["gap"].to_list() == [-50.0, 50.0, 150.0] and out["dist"].to_list() == [50.0, 50.0, 150.0]


def test_a_null_in_a_column_a_required_argument_reads_raises_like_a_function_step():
    nulls = pl.DataFrame({"amount": [1.0, None]})

    def cap(amount: float) -> float:
        return min(amount, 100.0)

    with pytest.raises(MissingInputError):
        flow(step(cap)).run(nulls)
    with pytest.raises(MissingInputError):
        _pipeline().run(nulls)


def test_missing_as_on_an_impl_argument_fills_a_null():
    class Floor(CustomFunction):
        x: str

        @staticmethod
        def impl(x: float = missing_as(0.0)) -> float:
            return x + 1.0

    out = assert_equivalent(flow(Floor(name="f", x="amount")), pl.DataFrame({"amount": [1.0, None]}))
    assert out["f"].to_list() == [2.0, 1.0]


def _slow(x: float) -> float:
    return x


def test_an_impl_numba_cannot_compile_falls_back_with_a_warning():
    class Opaque(CustomFunction):
        x: str

        @staticmethod
        def impl(x: float) -> float:
            return _slow(x)

    with pytest.warns(FallbackWarning):
        exe = flow(Opaque(name="o", x="amount"), name="p").compile("fused")
        assert exe.run(FRAME)["o"].to_list() == [50.0, 150.0, 250.0]
    assert set(exe.fallbacks()) == {"p/o"}


def test_allow_fallback_on_impl_silences_the_warning():
    class Accepted(CustomFunction):
        x: str

        @staticmethod
        @allow_fallback
        def impl(x: float) -> float:
            return _slow(x)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        exe = flow(Accepted(name="o", x="amount"), name="p").compile("fused")
        assert exe.run(FRAME)["o"].to_list() == [50.0, 150.0, 250.0]


def test_a_missing_impl_names_the_class_and_asks_for_a_staticmethod():
    with pytest.raises(TypeError, match=r"NoImpl must define `impl` as a @staticmethod"):
        class NoImpl(CustomFunction):
            x: str


def test_an_impl_that_is_not_a_staticmethod_is_rejected():
    with pytest.raises(TypeError, match=r"Method must define `impl` as a @staticmethod"):
        class Method(CustomFunction):
            x: str

            def impl(self, x: float) -> float:
                return x


def test_an_impl_argument_without_a_field_is_rejected_with_the_fix():
    with pytest.raises(TypeError, match=r"Orphan.impl has an argument 'y' but the class has no field 'y'"):
        class Orphan(CustomFunction):
            x: str

            @staticmethod
            def impl(x: float, y: float) -> float:
                return x + y


def test_a_field_that_is_neither_a_column_nor_a_value_is_rejected_with_the_alternatives():
    with pytest.raises(TypeError, match=r"Rows.rows is list.*param_table.*to_ir"):
        class Rows(CustomFunction):
            rows: list[float]

            @staticmethod
            def impl(rows: float) -> float:
                return rows
