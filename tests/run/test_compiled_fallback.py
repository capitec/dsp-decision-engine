"""Compiled modes run in Python, per step, whatever no kernel holds; @allow_fallback accepts it."""
import datetime as dt
import warnings

import polars as pl
import pytest

from decider.exceptions import FallbackWarning, WiringError

from decider import Raw, allow_fallback, flow, missing_as, raw_str, step
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


def test_string_steps_keep_their_semantics_and_run_in_python():
    with pytest.warns(FallbackWarning, match="p/pick runs in Python, row by row.*writes 'pick' as str"):
        out = assert_equivalent(flow(half, pick, same, is_complete, doubled, name="p"), VERSIONS)
    assert out["pick"].to_list() == ["v2", "v2", "v2"]
    assert out["same"].to_list() == [False, True, False]
    assert out["is_complete"].to_list() == [True, False, True]


def test_a_semantic_string_step_splits_the_fused_kernel_and_runs_in_python():
    from numba.core.dispatcher import Dispatcher

    exe = Engine().bind(flow(half, same, doubled, name="p"), mode="fused")
    with pytest.warns(FallbackWarning, match="p/same runs in Python, row by row"):
        exe.run(VERSIONS)
    units = [exe.runner.units[c.id] for c in exe.plan.calls]
    assert [type(u) for u in units] == [Kernel, Fallback, Kernel]
    # One compiled call per row costs more than the Python body it replaces, in a batch and on one record.
    assert not isinstance(units[1].fn, Dispatcher)


def test_the_warning_is_given_once_per_executable():
    exe = Engine().bind(flow(same, name="p"), mode="fused")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exe.run(VERSIONS)
        exe.run(VERSIONS)
        exe.score({"a_version": "v1", "b_version": "v1"})
    assert len(caught) == 1


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
    with pytest.warns(FallbackWarning, match="writes 'label' as str, which no kernel stores"):
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


@pytest.mark.parametrize("mode", ("interpreted", *COMPILED))
def test_raw_string_gives_the_same_answer_in_every_mode(mode):
    exe = Engine().bind(flow(is_raw_priority, name="p"), mode=mode)
    out = exe.run(pl.DataFrame({"value": ["priority", "other"]}))
    assert out["is_raw_priority"].to_list() == [True, False]


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


@pytest.mark.parametrize("mode", COMPILED)
def test_a_string_step_is_rejected_in_strict_mode_and_the_message_says_how_to_accept_it(mode):
    exe = Engine(strict_compile=True).bind(flow(is_priority, name="p"), mode=mode)
    with pytest.raises(ValueError, match=r"p/is_priority runs in Python, row by row.*@allow_fallback"):
        exe.score({"value": "priority"})


@allow_fallback
def declared_total(items: list[dict]) -> float:
    return sum(item["price"] for item in items)


@allow_fallback
def declared_sector(sector: str) -> float:
    return 1.0 if sector == "private" else 0.0


def priced(items: list[dict]) -> float:
    return sum(item["price"] for item in items)


DECLARED_STEP = allow_fallback(step(priced))


@pytest.mark.parametrize("fn, name, expected", [(declared_total, "declared_total", 2.0),
                                                (declared_sector, "declared_sector", 1.0),
                                                (DECLARED_STEP, "priced", 2.0)])
@pytest.mark.parametrize("mode", COMPILED)
def test_allow_fallback_is_silent_and_accepted_by_strict(mode, fn, name, expected):
    record = {"items": [{"price": 2.0}], "sector": "private"}
    exe = Engine(strict_compile=True).bind(flow(fn, name="p"), mode=mode)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert exe.score(record)[name] == expected
    assert exe.fallbacks()[f"p/{name}"].startswith("@allow_fallback: reads ")


@pytest.mark.parametrize("mode", COMPILED)
def test_fallback_warnings_are_silenced_by_their_category(mode):
    exe = Engine().bind(flow(order_total, name="order"), mode=mode)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=FallbackWarning)
        assert exe.score({"items": [{"price": 2.0}]})["order_total"] == 2.0


@pytest.mark.parametrize("bad", ["flow", "scorecard", "frame_step"])
def test_allow_fallback_refuses_a_step_it_could_not_reach(bad):
    from decider import frame_step
    from decider.steps.scorecard.config import DefaultBin, ScorecardConfig, ScoredVariable, ValuesBin

    targets = {
        "flow": lambda: flow(half, name="p"),
        "scorecard": lambda: ScorecardConfig(name="c", variables=[ScoredVariable(
            variable_name="s", bins=[ValuesBin(value=1.0, items=["v"])], default=DefaultBin(value=0.0))]),
        "frame_step": lambda: frame_step(reads=["x"], writes=["y"])(lambda df: df),
    }
    with pytest.raises(TypeError, match=r"takes a function, or step\(\) of one"):
        allow_fallback(targets[bad]())


def undeclared_identity(v):
    return v


def raw_code_through_a_helper(value: Raw[str]) -> bool:
    # No @helper, so the step can't compile: the `Raw[str]` code must still reach it.
    return undeclared_identity(value) == PRIORITY


def raw_span_through_a_helper(value: Raw[bytes]) -> bool:
    return undeclared_identity(value)[1] >= 0


@pytest.mark.parametrize("fn", [raw_code_through_a_helper, raw_span_through_a_helper])
def test_a_raw_input_keeps_its_representation_when_the_step_runs_in_python(fn):
    expected = {"raw_code_through_a_helper": [True, False], "raw_span_through_a_helper": [True, True]}
    with pytest.warns(FallbackWarning, match=f"p/{fn.__name__} runs in Python, row by row"):
        out = assert_equivalent(flow(fn, name="p"), pl.DataFrame({"value": ["priority", "other"]}))
    assert out[fn.__name__].to_list() == expected[fn.__name__]


def mixed_list(x: float) -> list[float]:
    return [1, 2] if x > 1 else [1.5]


@pytest.mark.parametrize("mode", COMPILED)
def test_a_numba_lowering_assert_falls_back_with_a_reason(mode):
    exe = Engine().bind(flow(mixed_list, name="p"), mode=mode)
    with pytest.warns(FallbackWarning, match="p/mixed_list runs in Python, row by row: AssertionError"):
        out = exe.run(pl.DataFrame({"x": [2.0, 0.5]}))
    assert out["mixed_list"].to_list() == [[1.0, 2.0], [1.5]]


def asserts_positive(x: float) -> float:
    assert x > 0.0
    return x


@pytest.mark.parametrize("mode", COMPILED)
def test_a_step_asserting_at_runtime_still_raises(mode):
    exe = Engine().bind(flow(asserts_positive, name="p"), mode=mode)
    assert exe.score({"x": 1.0})["asserts_positive"] == 1.0
    with pytest.raises(AssertionError):
        exe.score({"x": -1.0})
