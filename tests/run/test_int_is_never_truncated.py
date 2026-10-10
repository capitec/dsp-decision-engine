"""A float where an `int` is declared must be whole: 7.0 is accepted, 7.5, NaN and inf are refused, in every mode."""
import polars as pl
import pytest

from decikit import allow_fallback, flow
from decikit.exceptions import DecikitError, NotIntegralError
from decikit.steps.trees import TreeConfig

NOT_WHOLE = [7.5, float("nan"), float("inf"), -float("inf")]
HALVES_TO_A_FRACTION = [15.0, float("nan"), float("inf"), -float("inf")]


def to_int(x: int) -> int:
    return x


def half(y: float) -> int:
    return y / 2


def maybe_half(y: float) -> int | None:
    return y / 2


def source(y: float) -> float:
    return y


def reader(source: int) -> bool:
    return source > 0


def band_points(band: str) -> int:
    return 1.5 if band == "a" else 2


@allow_fallback
def declared_python(y: float) -> int:
    return y


def test_a_not_integral_error_is_a_decikit_value_error():
    assert issubclass(NotIntegralError, (DecikitError, ValueError))


@pytest.mark.parametrize("value", NOT_WHOLE)
def test_a_record_input_declared_int_refuses_a_float_that_is_not_whole(bind, value):
    exe = bind(flow(to_int, name="p"))
    with pytest.raises(NotIntegralError, match="input 'x'"):
        exe.score({"x": value})


@pytest.mark.parametrize("value", NOT_WHOLE)
def test_a_frame_input_declared_int_refuses_a_float_that_is_not_whole(bind, value):
    exe = bind(flow(to_int, name="p"))
    with pytest.raises(NotIntegralError, match="input 'x'"):
        exe.run(pl.DataFrame({"x": [1.0, value]}))


def test_a_whole_float_is_accepted_for_an_int_input_as_a_record_and_a_frame(bind):
    exe = bind(flow(to_int, name="p"))
    assert exe.score({"x": 7.0})["to_int"] == 7
    assert exe.run(pl.DataFrame({"x": [7.0, 8.0]}))["to_int"].to_list() == [7, 8]


@pytest.mark.parametrize("value", HALVES_TO_A_FRACTION)
def test_a_step_returning_a_float_that_is_not_whole_for_an_int_output_is_refused(bind, value):
    exe = bind(flow(half, name="p"))
    with pytest.raises(NotIntegralError, match="p/half"):
        exe.score({"y": value})
    with pytest.raises(NotIntegralError, match="p/half"):
        exe.run(pl.DataFrame({"y": [4.0, value]}))


def test_a_step_returning_a_whole_float_for_an_int_output_is_accepted(bind):
    exe = bind(flow(half, name="p"))
    assert exe.score({"y": 10.0})["half"] == 5
    assert exe.run(pl.DataFrame({"y": [4.0, 10.0]}))["half"].to_list() == [2, 5]


def test_an_optional_int_output_is_refused_too_and_a_null_is_not(bind):
    exe = bind(flow(maybe_half, name="p"))
    with pytest.raises(NotIntegralError, match="p/maybe_half"):
        exe.score({"y": 3.0})
    assert exe.score({"y": 4.0})["maybe_half"] == 2


def test_a_nan_from_a_float_step_into_an_int_output_is_refused_in_every_mode(bind):
    exe = bind(flow(source, half, name="p"))
    with pytest.raises(NotIntegralError):
        exe.score({"y": float("nan")})


@pytest.mark.filterwarnings("ignore::decikit.exceptions.FallbackWarning")
def test_a_python_run_step_returning_a_float_that_is_not_whole_for_an_int_output_is_refused(bind):
    exe = bind(flow(band_points, name="p"))
    assert exe.score({"band": "b"})["band_points"] == 2
    with pytest.raises(NotIntegralError, match="p/band_points"):
        exe.score({"band": "a"})
    with pytest.raises(NotIntegralError, match="p/band_points"):
        exe.run(pl.DataFrame({"band": ["b", "a"]}))


def test_an_allow_fallback_step_returning_a_fraction_for_an_int_output_is_refused(bind):
    exe = bind(flow(declared_python, name="p"))
    assert exe.score({"y": 3.0})["declared_python"] == 3
    with pytest.raises(NotIntegralError, match="p/declared_python"):
        exe.score({"y": 3.5})


def test_a_tree_int_feature_refuses_a_fraction_in_a_record_and_a_frame(bind):
    tree = TreeConfig(name="t", feature_types={"n": "int"}, tree={
        "nodes": [{"id": "root", "data": {"type": "unary", "condition": {"op": "==", "feature": "n", "threshold": 7}}},
                  {"id": "yes", "data": {"type": "leaf", "result_idx": 1}},
                  {"id": "no", "data": {"type": "leaf", "result_idx": 0}}],
        "edges": [{"source": "root", "target": "yes", "data": {"sourceIndex": 0}},
                  {"source": "root", "target": "no", "data": {"sourceIndex": 1}}],
        "output": {"data": [{"hit": 0}, {"hit": 1}], "default": {"hit": -1}, "dtypes": [["hit", "Int64"]]}})
    exe = bind(tree)
    assert exe.score({"n": 7.0})["hit"] == 1
    with pytest.raises(NotIntegralError, match="input 'n'"):
        exe.score({"n": 7.5})
    with pytest.raises(NotIntegralError, match="input 'n'"):
        exe.run(pl.DataFrame({"n": [7.0, 7.5]}))


def test_an_int_input_next_to_a_float_step_is_read_unchanged(bind):
    exe = bind(flow(source, reader, name="p"))
    assert exe.score({"y": 0.5})["reader"] is True
