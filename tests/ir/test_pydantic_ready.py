"""`pydantic_ready`: a `typing.TypedDict`, which pydantic refuses before Python 3.12, rebuilt so outputs and requests are still checked."""
from __future__ import annotations

import datetime as dt
from typing import Annotated, NotRequired, Optional, TypedDict

import pytest
from pydantic import Field, TypeAdapter, ValidationError

from decikit.ir import pydantic_ready


class Inner(TypedDict):
    day: dt.date
    count: NotRequired[int]


class Partial(TypedDict, total=False):
    items: list[Inner]
    maybe: Optional[Inner]
    positive: Annotated[float, Field(gt=0)]


class Outer(Partial):
    name: str


class Node(TypedDict):
    value: float
    children: list[Node]


def test_a_typing_typeddict_is_rebuilt_from_typing_extensions_at_any_depth():
    ready = pydantic_ready(list[Outer])
    outer = ready.__args__[0]
    assert type(outer).__module__ == "typing_extensions"
    assert outer.__required_keys__ == {"name"} and outer.__optional_keys__ == {"items", "maybe", "positive"}
    inner = outer.__annotations__["items"].__args__[0].__args__[0]
    assert type(inner).__module__ == "typing_extensions" and inner.__optional_keys__ == {"count"}


def test_the_rebuilt_type_validates_like_the_declared_one():
    adapter = TypeAdapter(list[pydantic_ready(Outer)])
    assert adapter.validate_python([{"name": "a", "items": [{"day": "2026-01-31"}], "maybe": None}]) == [
        {"name": "a", "items": [{"day": dt.date(2026, 1, 31)}], "maybe": None}]
    for bad in ({"items": []}, {"name": "a", "positive": -1.0}, {"name": "a", "items": [{"count": 1}]}):
        with pytest.raises(ValidationError):
            adapter.validate_python([bad])


def test_a_typeddict_that_holds_itself_is_left_as_declared_inside():
    assert pydantic_ready(Node).__annotations__["children"].__args__[0] is Node


def test_a_type_without_a_typeddict_is_returned_unchanged():
    assert pydantic_ready(float) is float
    assert pydantic_ready(list[dict[str, float]]) == list[dict[str, float]]
