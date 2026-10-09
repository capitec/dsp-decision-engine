"""A declared output that looks right is accepted without pydantic; everything else is judged by pydantic as before."""
from __future__ import annotations

from typing import NotRequired, Optional, TypedDict

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from decider._engine.compile.declared import _plain, output_check
from decider._engine.ir.decls import Output


class Row(TypedDict):
    a: float
    b: int
    c: NotRequired[str]


SHAPES = [list[int], list[float], list[bool], list[str], list[Optional[int]], Optional[list[int]],
          Row, list[Row], list[list[float]], list[Optional[Row]]]
VALUES = [None, [], [1, 2], [1.5, 2], [True, 1], [np.int64(3)], [np.float64(1.5)], ["a"], ["a", 1], [None, 1],
          [[1.0, 2], [3.0]], (1, 2), {"a": 1.5, "b": 2}, {"a": 1, "b": 2, "c": "x"}, {"a": 1, "b": 2.5},
          {"a": 1, "b": True}, {"a": 1}, {"a": 1, "b": 2, "z": 9}, {"a": 1, "b": 2, "c": None},
          [{"a": 1, "b": 2}, None], [{"a": 1.0, "b": 2}], np.int64(1), 1, "x"]


def strict(shape, value) -> bool:
    try:
        TypeAdapter(Optional[shape]).validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


def accepts(shape, value) -> bool:
    # What a step's output is held to: strict, after a numpy number or `Text` is read as the Python value it is.
    return strict(shape, value) or strict(shape, _plain(value))


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_the_fast_path_accepts_only_what_strict_pydantic_accepts(shape):
    fast = output_check(Output("out", shape))._fast
    assert fast is not None
    for value in VALUES:
        if fast(value):
            assert strict(shape, value), value


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_a_value_pydantic_refuses_is_still_refused_naming_step_and_output(shape):
    check = output_check(Output("out", shape))
    for value in VALUES:
        if accepts(shape, value):
            check("s", Output("out", shape), [value])
        else:
            with pytest.raises(ValueError, match="step s: output 'out'"):
                check("s", Output("out", shape), [value])


def test_a_numpy_number_in_a_list_is_accepted_by_the_plain_pass():
    check = output_check(Output("out", list[int]))
    check("s", Output("out", list[int]), [[np.int64(3), 4]])
