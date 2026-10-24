"""A step annotated Decimal or Currency: one answer in every mode, int64 everywhere it is stored."""
import decimal as stdlib
from typing import Annotated

import numpy as np
import polars as pl
import pytest
from numba.core import types

from decikit import Engine, FusedEngine, InterpretedEngine, SteppedEngine, allow_fallback, branch, flow, loop, missing_as, param, step
from decikit._engine.compile import default_bundle
from decikit._engine.compile.decimal_type import DecimalType
from decikit._engine.compile.njit import _input_type
from decikit.exceptions import IRError, ParamsError, WiringError
from decikit.ir import to_ir
from decikit.ops import decimal
from decikit.ops.decimal import Currency, Decimal, Rounding
from decikit.lab.testing import assert_equivalent, corpus

MODES = ("interpreted", "stepped", "fused")
D = stdlib.Decimal


def instalment(principal: Currency["R"], rate: float = param(0.1275), months: int = param(36)) -> Currency["R"]:
    return decimal.round(principal * rate / months, 2, Rounding.HALF_EVEN)


def with_fee(instalment: Currency["R"], floor: Currency = param(Currency("12.50"), ge=0)) -> Currency["R"]:
    return max(instalment, floor) + Currency(1, 2500)


MONEY = flow(instalment, with_fee, name="loan")
FRAME = pl.DataFrame({"principal": [D("1000.50"), D("25.00"), D("0.00"), None]}, schema={"principal": pl.Decimal(18, 2)}).drop_nulls()


def test_a_money_flow_agrees_in_every_mode_over_the_corpus_and_a_frame():
    assert_equivalent(MONEY, FRAME)
    for frame in corpus(MONEY).values():
        assert_equivalent(MONEY, frame)


@pytest.mark.parametrize("mode", MODES)
def test_the_answer_is_exact_decimal_arithmetic_in_every_mode(mode):
    out = Engine.resolve(mode)().bind(MONEY).run(FRAME)
    # 1000.50 * 0.1275 / 36 = 3.54343...; the floor is 12.50; the fee adds 1.25
    assert out["with_fee"].dtype == pl.Decimal(38, 4)
    assert out["with_fee"].to_list() == [D("13.7500"), D("13.7500"), D("13.7500")]
    assert out["principal"].dtype == pl.Decimal(18, 2) and out["principal"].to_list() == FRAME["principal"].to_list()


@pytest.mark.parametrize("mode", ("stepped", "fused"))
def test_a_decimal_step_compiles_with_no_fallback(mode):
    exe = Engine.resolve(mode)(strict_compile=True).bind(MONEY)
    exe.run(FRAME)
    assert exe.run({"principal": "1000.50"})["with_fee"] == Currency("13.75")
    assert exe.fallbacks() == {}


# ---- run() on a record takes anything the construction rules take, and gives back Decimal or Currency

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("value", [Currency("2000.5"), Decimal("2000.5000"), D("2000.5"), 2000.5, "2000.5", Decimal("2000.5", units="$")])
def test_a_record_run_accepts_every_way_to_write_an_amount(mode, value):
    out = Engine.resolve(mode)().bind(flow(instalment)).run({"principal": value})
    assert out["instalment"] == Currency("7.09") and type(out["instalment"]) is Currency
    assert out["instalment"].units == "R" and repr(out["instalment"]) == "R 7.0900"


def test_a_record_run_rounds_an_input_to_the_declared_scale_half_even():
    exe = FusedEngine().bind(flow(passthrough))
    assert exe.run({"x": "2.675"})["passthrough"] == Decimal("2.68")
    assert exe.run({"x": 2.675})["passthrough"] == Decimal("2.67")    # the double is 2.67499999999999982...
    assert exe.run({"x": 3})["passthrough"] == Decimal("3.00")


def passthrough(x: Decimal[10, 2]) -> Decimal[10, 2]:
    return x


def test_a_record_run_refuses_a_value_that_is_not_a_number():
    exe = FusedEngine().bind(flow(passthrough))
    for bad in ("abc", None, [1]):
        with pytest.raises((ValueError, TypeError)):
            exe.run({"x": bad})
    with pytest.raises(OverflowError):
        exe.run({"x": "9999999999999999999"})


# ---- the boundary

def test_a_column_of_another_scale_is_rescaled_half_even_and_an_integer_column_is_whole_units():
    frame = pl.DataFrame({"x": [D("2.675"), D("2.665"), D("-2.675")]}, schema={"x": pl.Decimal(10, 3)})
    out = FusedEngine().bind(flow(passthrough)).run(frame)
    assert out["passthrough"].to_list() == [D("2.68"), D("2.66"), D("-2.68")]
    out = FusedEngine().bind(flow(passthrough)).run(pl.DataFrame({"x": [3, -4]}))
    assert out["passthrough"].to_list() == [D("3.00"), D("-4.00")]


def test_a_decimal_column_too_wide_for_int64_is_an_error_naming_the_column():
    frame = pl.DataFrame({"x": [D("9" * 30)]}, schema={"x": pl.Decimal(38, 0)})
    with pytest.raises(Exception, match=r"column 'x'.*does not fit a Decimal at scale 2"):
        FusedEngine().bind(flow(passthrough)).run(frame)


def test_nulls_in_and_out_of_a_decimal_column_follow_the_null_policy():
    def maybe(x: Decimal[10, 2] | None, y: Currency = missing_as(Currency(3))) -> Currency | None:
        return None if x is None else x + y

    frame = pl.DataFrame({"x": [D("1.5"), None], "y": [None, D("2")]}, schema={"x": pl.Decimal(10, 2), "y": pl.Decimal(10, 1)})
    out = assert_equivalent(flow(maybe), frame)
    assert out["maybe"].to_list() == [D("4.5000"), None]
    assert FusedEngine().bind(flow(maybe)).run({"x": None})["maybe"] is None


# ---- stored as int64, typed as a decimal only inside a step

def test_no_decimal_object_reaches_a_dispatcher_on_the_warm_path():
    node = next(n for n in _calls(to_ir(MONEY)) if n.origin.path.endswith("with_fee"))
    bundle = default_bundle(node)
    assert bundle == (125000,) and type(bundle[0]) is int
    assert _input_type(node.inputs[0]) == DecimalType(4)
    exe = FusedEngine().bind(MONEY)
    state, _ = exe.prepare(FRAME)
    assert {v.name: a.dtype for v, a in ((v, state.values[v.id]) for v in exe.plan.versions if v.id in state.values)}["principal"] == np.int64


def _calls(node):
    from decikit.ir import CallNode, iter_nodes

    return (n for n in iter_nodes(node) if isinstance(n, CallNode))


def test_an_optional_decimal_input_is_an_optional_decimal_in_the_kernel():
    def maybe(x: Currency | None) -> Currency | None:
        return x

    node = next(_calls(to_ir(flow(maybe))))
    assert _input_type(node.inputs[0]) == types.Optional(DecimalType(4))


# ---- output scale

def test_a_coarser_output_rounds_half_even_and_an_int_return_is_whole_units():
    def fine(a: Currency) -> Decimal[10, 2]:
        return a / 8

    def whole(a: Currency) -> Decimal[10, 2]:
        return 7

    frame = pl.DataFrame({"a": [D("1"), D("3"), D("-1")]}, schema={"a": pl.Decimal(10, 4)})
    out = assert_equivalent(flow(fine, whole), frame)     # 0.125, 0.375, -0.125
    assert out["fine"].to_list() == [D("0.12"), D("0.38"), D("-0.12")]
    assert out["whole"].to_list() == [D("7.00")] * 3


@pytest.mark.parametrize("mode", MODES)
def test_a_float_returned_for_a_decimal_output_is_a_type_error_in_every_mode(mode):
    def bad(a: Currency) -> Currency:
        return float(a) * 0.5

    with pytest.raises(TypeError, match="Decimal"):
        Engine.resolve(mode)().bind(flow(bad)).run({"a": 1})


# ---- a wrong scale between steps is a wiring error, not a silent misread

def test_a_step_reading_a_value_at_another_scale_is_refused():
    def fine(a: Decimal[10, 2]) -> Decimal[10, 4]:
        return a

    def coarse(fine: Decimal[10, 2]) -> Decimal[10, 2]:
        return fine

    with pytest.raises(WiringError, match=r"reads 'fine' as Decimal at scale 2, but .* writes it as Decimal at scale 4"):
        InterpretedEngine().bind(flow(fine, coarse))
    with pytest.raises(WiringError, match=r"as float.*writes it as Currency at scale 4"):
        def plain(fine: float) -> float:
            return fine

        def money(a: Currency) -> Currency:
            return a

        def reader(money: float) -> float:
            return money

        InterpretedEngine().bind(flow(money, reader))


def test_two_steps_reading_one_input_column_at_different_scales_are_refused():
    def a(x: Decimal[10, 2]) -> int:
        return 1

    def b(x: Decimal[10, 4]) -> int:
        return 2

    with pytest.raises(WiringError, match="an input column has one type"):
        InterpretedEngine().bind(flow(a, b))


# ---- annotations

def test_a_decimal_annotation_needs_a_precision_and_scale_and_at_most_nineteen_digits():
    def bare(x: Decimal) -> int:
        return 1

    with pytest.raises(IRError, match=r"write Decimal\[18, 2\]"):
        flow(bare)
    with pytest.raises(ValueError, match="at most 19"):
        Decimal[20, 2]
    with pytest.raises(ValueError, match="scale must be"):
        Decimal[10, 11]
    with pytest.raises(TypeError, match=r"Decimal\[precision, scale\]"):
        Decimal[18]


def test_a_too_wide_annotation_raises_at_flow_build_when_annotations_are_strings():
    ns: dict = {}
    exec("from __future__ import annotations\nfrom decikit.ops.decimal import Decimal\n"
         "def wide(x: Decimal[38, 2]) -> int:\n    return 1\n", ns)
    with pytest.raises(IRError, match="at most 19"):
        flow(ns["wide"])


def test_currency_is_decimal_19_4_and_decimal_spells_its_own_precision():
    from decikit._engine.types import decimal_info

    assert decimal_info(Currency)[1:3] == (4, None) and decimal_info(Currency["R"])[1:3] == (4, "R")
    assert decimal_info(Decimal[18, 2]).scale == 2 and decimal_info(Decimal[18, 2] | None).scale == 2
    assert decimal_info(Annotated[int, "x"]) is None and decimal_info(float) is None


# ---- params

def test_a_decimal_param_takes_a_number_or_string_validates_bounds_and_stores_an_int64():
    exe = FusedEngine().bind(MONEY)
    row = {"principal": "100"}
    base = exe.run(row)["with_fee"]
    assert base == Currency("13.75")
    assert exe.run(row, {"loan": {"with_fee": {"floor": "20"}}})["with_fee"] == Currency("21.25")
    assert exe.run(row, {"loan": {"with_fee": {"floor": 20.5}}})["with_fee"] == Currency("21.75")
    with pytest.raises(ParamsError, match="floor"):
        exe.run(row, {"loan": {"with_fee": {"floor": "-1"}}})
    with pytest.raises(ParamsError, match="floor"):
        exe.run(row, {"loan": {"with_fee": {"floor": "abc"}}})


def test_the_params_document_shows_a_decimal_default_as_its_digits():
    schema = MONEY.parameters()
    assert schema.defaults()["loan"]["with_fee"] == {"floor": "12.5000"}
    assert schema["loan/with_fee"]["floor"]["type"] == "Currency" and schema["loan/with_fee"]["floor"]["default"] == "12.5000"
    assert schema.json_schema()["properties"]["loan"]["properties"]["with_fee"]["properties"]["floor"]["default"] == "12.5000"


@pytest.mark.parametrize("mode", MODES)
def test_a_decimal_param_and_input_reach_a_python_step_as_decimals(mode):
    @allow_fallback
    def seen(x: Currency, floor: Currency = param(Currency(5))) -> int:
        # `.scale` exists only on the Python object: no kernel takes this step.
        return floor.scale * 10 + x.scale + (1 if floor == Currency(5) else 0)

    assert Engine.resolve(mode)().bind(flow(seen)).run({"x": 1})["seen"] == 45


# ---- constructing values in a step

@pytest.mark.parametrize("mode", MODES)
def test_a_step_builds_currency_and_decimal_values(mode):
    def build(n: int) -> Currency:
        base = Currency(2, 6500) + Currency(n) + Decimal(1, 2)
        return -base if n > 5 else abs(-base)

    exe = Engine.resolve(mode)(strict_compile=True).bind(flow(build))
    assert exe.run({"n": 1})["build"] == Currency("4.65")
    assert exe.run({"n": 6})["build"] == Currency("-9.65")


def test_currency_major_minor_follows_its_rules_in_a_step():
    def build(major: int, minor: int) -> Currency:
        return Currency(major, minor)

    exe = FusedEngine().bind(flow(build))
    assert exe.run({"major": -2, "minor": 5})["build"] == Currency("-2.0005")
    assert exe.run({"major": 0, "minor": 9999})["build"] == Currency("0.9999")
    with pytest.raises(ValueError, match="minor"):
        exe.run({"major": 1, "minor": 10000})


def test_negation_in_python_keeps_a_currency_and_its_units():
    value = -Currency(2, 6500, units="R")
    assert type(value) is Currency and repr(value) == "R -2.6500"


# ---- control flow

def positive(balance: Currency) -> bool:
    return balance > 0


@step(output="balance")
def pay(balance: Currency, payment: Currency = param(Currency(100)), rate: float = param(0.01)) -> Currency:
    return balance * (1.0 + rate) - payment


@step(output="months")
def tick(months: int) -> int:
    return months + 1


def over_cap(size: Currency, cap: Currency = param(Currency(500))) -> bool:
    return size > cap


@step(output="balance")
def discount(balance: Currency) -> Currency:
    return balance * 0.9


def test_a_decimal_loop_and_branch_pack_into_kernels_and_agree_with_python():
    repay = loop(positive, flow(pay, tick, name="month"), carries=["balance", "months"], max_iterations=60, name="repay")
    pipeline = flow(branch(over_cap, discount, modifies=["balance"], name="big"), repay, name="plan")
    frame = pl.DataFrame({"size": [D("1000"), D("250.25"), D("0")], "balance": [D("1000"), D("250.25"), D("0")], "months": [0, 0, 0]},
                         schema={"size": pl.Decimal(18, 4), "balance": pl.Decimal(18, 4), "months": pl.Int64})
    out = assert_equivalent(pipeline, frame)
    assert out["months"].to_list() == [10, 3, 0]
    exe = FusedEngine(strict_compile=True).bind(pipeline)
    exe.run(frame)
    assert sorted(exe.runner.packed) == ["plan/big", "plan/repay"]


# ---- debugging

def test_a_session_override_of_a_decimal_takes_any_way_to_write_an_amount():
    exe = SteppedEngine().bind(flow(instalment, with_fee, name="loan"))
    s = exe.session(FRAME)
    s.break_at("loan/with_fee")
    s.resume()
    s.set("instalment", "20")
    s.resume()
    assert s.output()["with_fee"].to_list()[0] == D("21.2500")
    assert s.value("instalment").to_list()[0] == D("20.0000")
