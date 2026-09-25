"""Compiled modes run unsupported values in Python and semantic strings in kernels."""
import datetime as dt
import warnings

import polars as pl
import pytest

from decider.exceptions import WiringError

from decider import Raw, flow, missing_as, raw_str
from decider.engine import Engine
from decider.engine.compile import Fallback, Kernel
from decider.testing import assert_equivalent

COMPILED = ("stepped", "fused")


def half(x: float) -> float:
    return x / 2


def pick(a_version: str, b_version: str, half: float) -> str:
    return a_version if half > 1 else b_version


def same(a_version: str, b_version: str) -> bool:
    return a_version == b_version


def is_complete(band: str) -> bool:
    return band == "complete"


def doubled(half: float) -> float:
    return half * 2


VERSIONS = pl.DataFrame({"x": [1.0, 4.0, 6.0], "a_version": ["v1", "v2", "v2"], "b_version": ["v2", "v2", "v1"],
                         "band": ["complete", "partial", "complete"]})


def test_string_steps_preserve_semantics_and_fallback_only_for_string_outputs():
    with pytest.warns(UserWarning, match="p/pick runs in Python.*writes 'pick' as str"):
        out = assert_equivalent(flow(half, pick, same, is_complete, doubled, name="p"), VERSIONS)
    assert out["pick"].to_list() == ["v2", "v2", "v2"]
    assert out["same"].to_list() == [False, True, False]
    assert out["is_complete"].to_list() == [True, False, True]


def test_semantic_string_steps_share_fused_kernel():
    exe = Engine().bind(flow(half, same, doubled, name="p"), mode="fused")
    exe.run(VERSIONS)
    units = list({id(u): u for u in exe.runner.units.values()}.values())
    assert [type(u) for u in units] == [Kernel]


def test_the_warning_is_given_once_per_executable():
    exe = Engine().bind(flow(same, name="p"), mode="fused")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exe.run(VERSIONS)
        exe.run(VERSIONS)
        exe.score({"a_version": "v1", "b_version": "v1"})
    assert len(caught) == 0


def as_dict(x: float) -> dict:
    return {"a": x}


def as_list(x: float) -> list[float]:
    return [x, x]


def as_date(x: float) -> dt.date:
    return dt.date(2026, 1, int(x))


@pytest.mark.parametrize("mode", COMPILED)
@pytest.mark.parametrize("fn", [as_dict, as_list, as_date])
def test_a_python_fallback_keeps_object_outputs_as_python_values(mode, fn):
    expected = Engine().bind(flow(fn, name="p")).score({"x": 3.0})
    assert Engine().bind(flow(fn, name="p"), mode=mode).score({"x": 3.0}) == expected


def days_to(decision_date: dt.date, x: float) -> float:
    return x


def checked(x: float) -> float:
    # A generator in any(): numba on Python 3.14 can't read its bytecode.
    return x + 1 if any(x > t for t in (0.0, 1.0)) else x


def test_a_step_reading_a_date_or_numba_cannot_read_runs_in_python():
    df = pl.DataFrame({"decision_date": [dt.date(2026, 1, 1)], "x": [2.0]})
    out = assert_equivalent(flow(days_to, checked, name="p"), df)
    assert out["days_to"].to_list() == [2.0] and out["checked"].to_list() == [3.0]
    exe = Engine().bind(flow(days_to, name="p"), mode="stepped")
    exe.run(df)
    assert "reads 'decision_date'" in exe.runner.units[exe.plan.calls[0].id].reason


def as_float(code: float) -> float:
    return code / 2


def as_int(code: int) -> int:
    return code // 2


def test_an_input_read_as_int_and_as_float_is_refused_with_a_fix():
    with pytest.raises(WiringError, match="reads input column 'code' as int, but p/as_float reads it as float"):
        Engine().bind(flow(as_int, as_float, name="p"), mode="fused")


def total(hist: list[float] = missing_as([])) -> float:
    return float(sum(hist))


@pytest.mark.parametrize("mode", COMPILED)
def test_an_empty_list_fill_fills_each_missing_row(mode):
    exe = Engine().bind(flow(total, name="p"), mode=mode)
    assert exe.score({"x": 1.0})["total"] == 0.0
    out = exe.run(pl.DataFrame({"hist": [[1.0, 2.0], None]}, schema={"hist": pl.List(pl.Float64)}))
    assert out["total"].to_list() == [3.0, 0.0]


def order_total(items: list[dict]) -> float:
    return sum(item["price"] for item in items)


def is_priority(value: str) -> bool:
    return value == "priority"


def is_binary_priority(value: bytes) -> bool:
    return value == b"priority"


def has_raw_string(value: Raw[str]) -> bool:
    return value >= 0


PRIORITY = raw_str("priority")


def is_raw_priority(value: Raw[str]) -> bool:
    return value == PRIORITY


def has_raw_bytes(value: Raw[bytes]) -> bool:
    return value[1] >= 0


@pytest.mark.parametrize("mode", COMPILED)
def test_scalar_string_steps_receive_semantic_strings(mode):
    exe = Engine().bind(flow(is_priority, name="p"), mode=mode)
    out = exe.run(pl.DataFrame({"value": ["priority", "other"]}))
    assert out["is_priority"].to_list() == [True, False]


@pytest.mark.parametrize("mode", COMPILED)
def test_scalar_semantic_bytes_fall_back_until_adapter_exists(mode):
    exe = Engine().bind(flow(is_binary_priority, name="p"), mode=mode)
    with pytest.warns(UserWarning, match="reads 'value' as bytes: numba can't type a scalar bytes value"):
        out = exe.run(pl.DataFrame({"value": [b"priority", b"other"]}))
    assert out["is_binary_priority"].to_list() == [True, False]


def label(x: float) -> str:
    return "big" if x > 1 else "small"


def label_code(x: float) -> Raw[str]:
    return PRIORITY if x > 1 else 0


@pytest.mark.parametrize("mode", COMPILED)
def test_str_output_falls_back_but_raw_str_output_joins_the_kernel(mode):
    exe = Engine().bind(flow(label, name="p"), mode=mode)
    with pytest.warns(UserWarning, match="writes 'label' as str, which no fused kernel stores"):
        out = exe.run(pl.DataFrame({"x": [2.0, 0.5]}))
    assert out["label"].to_list() == ["big", "small"]

    exe = Engine().bind(flow(label_code, name="p"), mode=mode)
    out = exe.run(pl.DataFrame({"x": [2.0, 0.5]}))
    assert out["label_code"].to_list() == [PRIORITY, 0]
    assert exe.fallbacks() == {}


@pytest.mark.parametrize("fn", [has_raw_string, has_raw_bytes])
@pytest.mark.parametrize("mode", COMPILED)
def test_raw_annotations_use_internal_representations(mode, fn):
    exe = Engine().bind(flow(fn, name="p"), mode=mode)
    out = exe.run(pl.DataFrame({"value": ["priority", "other"]}))
    assert out[fn.__name__].to_list() == [True, True]


@pytest.mark.parametrize("mode", COMPILED)
def test_raw_string_constants_are_preconverted(mode):
    exe = Engine().bind(flow(is_raw_priority, name="p"), mode=mode)
    out = exe.run(pl.DataFrame({"value": ["priority", "other"]}))
    assert out["is_raw_priority"].to_list() == [True, False]


@pytest.mark.parametrize("mode", COMPILED)
def test_list_input_fallback_warns_and_is_public(mode):
    exe = Engine().bind(flow(order_total, name="order"), mode=mode)
    with pytest.warns(UserWarning, match="order/order_total runs in Python, row by row"):
        assert exe.score({"items": [{"price": 2.0}, {"price": 3.0}]})["order_total"] == 5.0
    assert "reads 'items' as list[dict], which no kernel takes" in exe.fallbacks()["order/order_total"]


@pytest.mark.parametrize("mode", COMPILED)
def test_list_input_fallback_is_rejected_in_strict_mode(mode):
    exe = Engine(strict_compile=True).bind(flow(order_total, name="order"), mode=mode)
    with pytest.raises(ValueError, match="order/order_total runs in Python, row by row"):
        exe.score({"items": [{"price": 2.0}]})
