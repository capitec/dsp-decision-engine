"""A step's dict, TypedDict and list outputs are held to the type it declares, in every mode."""
from __future__ import annotations

import json
import math
import warnings
from typing import NotRequired, TypedDict

import numpy as np
import polars as pl
import pytest

from decikit import Engine, allow_fallback, flow
from decikit.config import JsonFileStore
from decikit.exceptions import FallbackWarning
from decikit.ops import Columnar, Struct
from decikit.serving import RequestHandler

MODES = ("interpreted", "stepped", "fused")


class Offer(TypedDict):
    v: float


class Tagged(TypedDict):
    v: float
    note: NotRequired[str]


class Pair(TypedDict):
    lo: float
    hi: float


def bound(fn, mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FallbackWarning)
        return Engine().bind(flow(fn, name="p"), mode=mode)


def labelled(x: float) -> Offer:
    return {"v": "high" if x > 1 else x}


def constant_text(x: float) -> Offer:
    return {"v": "high"}


def one_bad_in_list(x: float) -> list[Offer]:
    return [{"v": 1.0}, {"v": "high"}]


def widened(x: float) -> Offer:
    return {"v": 3}


def bool_for_float(x: float) -> Offer:
    return {"v": True}


def optional_note(x: float) -> Tagged:
    return {"v": x}


@allow_fallback
def declared_fallback(x: float) -> Offer:
    return {"v": "high"}


def pair(x: float) -> Struct[Pair]:
    return {"lo": x, "hi": "up"}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fn", [constant_text, one_bad_in_list, declared_fallback])
def test_a_wrong_value_in_a_declared_dict_output_is_refused_naming_step_and_field(mode, fn):
    exe = bound(fn, mode)
    with pytest.raises(ValueError, match=rf"step p/{fn.__name__}: output '{fn.__name__}' row 0.*v.*'high'"):
        exe.run({"x": 1.0})


@pytest.mark.parametrize("mode", MODES)
def test_a_frame_run_refuses_a_wrong_value_instead_of_turning_numbers_into_text(mode):
    exe = bound(labelled, mode)
    with pytest.raises(ValueError, match=r"step p/labelled: output 'labelled' row 1, field v.*'high'"):
        exe.run(pl.DataFrame({"x": [0.5, 2.0]}))


class Item(TypedDict):
    price: float
    count: int


def tuple_struct(x: float) -> Struct[Item]:
    return (x, x)


def good_struct(x: float) -> Struct[Item]:
    return (x, 1)


def tuple_rows(x: float) -> Columnar[Item]:
    return [(x, 1), (x, 2.5)]


def good_rows(x: float) -> Columnar[Item]:
    return [(x, 1), (x + 1, 2)]


def short_tuple_rows(x: float) -> Columnar[Item]:
    return [(x,)]


@pytest.mark.parametrize("fn, field", [(tuple_struct, "count"), (tuple_rows, "count"), (short_tuple_rows, "")])
def test_a_tuple_output_that_cannot_be_the_declared_record_is_reported_at_compile_time(fn, field):
    exe = Engine().bind(flow(fn, name="p"), mode="fused")
    with pytest.warns(FallbackWarning, match=f"p/{fn.__name__} runs in Python"):
        with pytest.raises(ValueError, match=f"step p/{fn.__name__}: output '{fn.__name__}' row 0"):
            exe.run({"x": 1.0})
    assert fn.__name__ in " ".join(exe.fallbacks())
    if field:
        assert f"field '{field}'" in exe.fallbacks()[f"p/{fn.__name__}"]


@pytest.mark.parametrize("fn", [good_struct, good_rows])
def test_a_tuple_output_matching_the_declared_record_stays_in_the_kernel(fn):
    exe = Engine(strict_compile=True).bind(flow(fn, name="p"), mode="fused")
    assert exe.run({"x": 1.0})[fn.__name__] is not None
    assert exe.fallbacks() == {}


@pytest.mark.parametrize("mode", MODES)
def test_an_int_is_a_valid_float_but_a_bool_is_not(mode):
    assert bound(widened, mode).run({"x": 1.0})["widened"] == {"v": 3}
    with pytest.raises(ValueError, match="bool_for_float.*field v"):
        bound(bool_for_float, mode).run({"x": 1.0})


@pytest.mark.parametrize("mode", MODES)
def test_a_missing_not_required_key_is_accepted(mode):
    assert bound(optional_note, mode).run({"x": 1.0})["optional_note"] == {"v": 1.0}


def test_a_struct_output_is_held_to_its_fields():
    with pytest.raises(ValueError, match="output 'pair' row 0, field hi"):
        bound(pair, "interpreted").run({"x": 1.0})


def mixed_rows(x: float) -> list[Offer]:
    return [{"v": int(x)}] if x < 2 else [{"v": x}]


@pytest.mark.parametrize("mode", MODES)
def test_a_typed_list_with_an_int_in_one_row_and_a_float_in_the_next_keeps_the_float(mode):
    out = bound(mixed_rows, mode).run(pl.DataFrame({"x": [1.0, 2.5]}))
    assert out["mixed_rows"].to_list() == [[{"v": 1.0}], [{"v": 2.5}]]


def untyped(x: float) -> dict:
    return {"v": "high" if x > 1 else x}


def untyped_numbers(x: float) -> dict:
    return {"v": int(x) if x > 1 else x}


def int_keys(x: float) -> dict:
    return {1: x}


@allow_fallback
def int_keys_in_list(x: float) -> list[dict]:
    return [{1: x}]


@pytest.mark.parametrize("mode", MODES)
def test_an_undeclared_dict_with_a_number_in_one_row_and_text_in_the_next_is_refused(mode):
    with pytest.raises(ValueError, match="output 'untyped', field v.*number and str"):
        bound(untyped, mode).run(pl.DataFrame({"x": [0.5, 2.0]}))


@pytest.mark.parametrize("mode", MODES)
def test_an_undeclared_dict_widens_an_int_beside_a_float(mode):
    out = bound(untyped_numbers, mode).run(pl.DataFrame({"x": [0.5, 2.0]}))
    assert out["untyped_numbers"].to_list() == [{"v": 0.5}, {"v": 2.0}]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fn, key", [(int_keys, "int_keys"), (int_keys_in_list, "int_keys_in_list")])
def test_a_frame_run_turns_non_string_dict_keys_into_text_and_a_record_run_keeps_them(mode, fn, key):
    exe = bound(fn, mode)
    out = exe.run(pl.DataFrame({"x": [0.5]}))[key].to_list()[0]
    assert out in ({"1": 0.5}, [{"1": 0.5}])
    assert exe.run({"x": 0.5})[key] in ({1: 0.5}, [{1: 0.5}])


class Count(TypedDict):
    n: int
    share: float
    flag: bool


def numpy_numbers(x: float) -> list[Count]:
    return [{"n": np.int64(3), "share": np.float32(0.5), "flag": np.bool_(x > 1)}]


@pytest.mark.parametrize("mode", MODES)
def test_a_numpy_number_in_a_declared_dict_is_the_number_it_is(mode):
    exe = bound(numpy_numbers, mode)
    assert exe.run({"x": 2.0})["numpy_numbers"] == [{"n": 3, "share": 0.5, "flag": True}]
    assert exe.run(pl.DataFrame({"x": [0.5, 2.0]}))["numpy_numbers"].to_list() == [
        [{"n": 3, "share": 0.5, "flag": False}], [{"n": 3, "share": 0.5, "flag": True}]]


def test_a_numpy_number_in_a_dict_output_is_served_as_json(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path / "configs"))
    store.create_version({"params": {}})
    handler = RequestHandler(store, lambda: flow(numpy_numbers, name="p"))
    handler.stage()
    handler.activate()
    body = handler.process_fn(b'{"x": 2.0}', "application/json", "application/json").content
    assert json.loads(body)["numpy_numbers"] == [{"n": 3, "share": 0.5, "flag": True}]


def counted(x: float) -> list[Count]:
    return [{"n": 1, "share": 0.5, "flag": True}, {"n": x, "share": 0.5, "flag": True}]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("x", [7.5, 7.0, math.nan])
def test_a_float_in_an_int_field_of_a_declared_list_is_refused_naming_the_field(mode, x):
    with pytest.raises(ValueError, match=r"output 'counted' row 0, field 1\.n: .*valid integer"):
        bound(counted, mode).run({"x": x})


def numpy_in_one_row(x: float) -> dict:
    return {"n": np.int64(3) if x > 1 else 4}


@pytest.mark.parametrize("mode", MODES)
def test_an_undeclared_dict_counts_a_numpy_number_as_a_number(mode):
    assert bound(numpy_in_one_row, mode).run(pl.DataFrame({"x": [0.5, 2.0]}))["numpy_in_one_row"].to_list() == [
        {"n": 4}, {"n": 3}]
