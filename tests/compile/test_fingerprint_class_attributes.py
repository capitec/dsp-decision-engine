"""An object held as a class attribute is hidden state like a module global; class machinery is not."""
import abc
import dataclasses
import enum
import functools
import typing

import pydantic
import pytest

from decikit import Engine, HiddenStateWarning, flow
from decikit._engine.compile import fingerprint, hidden_state
from decikit.exceptions import EngineError


class Register:
    def __init__(self, cap):
        self.c = cap

    def cap(self, x):
        return min(x, self.c)


class Holder:
    reg = Register(5.0)


class Outer:
    class Inner:
        reg = Register(5.0)


def capped(x: float) -> float:
    return Holder.reg.cap(x)


def nested(x: float) -> float:
    return Outer.Inner.reg.cap(x)


def test_an_object_class_attribute_is_named_with_the_class():
    [(name, kind, type_name, site)] = hidden_state(capped)
    assert (name, kind, type_name) == ("Holder.reg", "object", "Register")
    assert site.startswith("module global in ")
    assert [h[:3] for h in hidden_state(nested)] == [("Outer.Inner.reg", "object", "Register")]


@pytest.mark.parametrize("fn", [capped, nested])
def test_an_object_class_attribute_is_refused_by_the_strict_key(fn):
    with pytest.raises(ValueError, match="Register object has no content key"):
        fingerprint(fn, strict=True)


def test_the_default_key_of_an_object_class_attribute_does_not_depend_on_the_objects_identity():
    first = fingerprint(capped)
    Holder.reg = Register(5.0)
    assert fingerprint(capped) == first


def test_binding_warns_and_strict_compile_raises_on_an_object_class_attribute():
    with pytest.warns(HiddenStateWarning, match=r"p/capped reads Register `Holder.reg`"):
        Engine.resolve("interpreted")().bind(flow(capped, name="p"))
    with pytest.raises(EngineError, match=r"p/capped reads Register `Holder.reg`"):
        Engine.resolve("fused")(strict_compile=True).bind(flow(capped, name="p"))


class Model(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
    rate: float = 1.0
    tags: list[str] = pydantic.Field(default_factory=list)

    @pydantic.field_validator("rate")
    @classmethod
    def positive(cls, v):
        return v

    @pydantic.computed_field
    @property
    def double(self) -> float:
        return self.rate * 2


@dataclasses.dataclass(frozen=True)
class Terms:
    rate: float = 1.0
    tags: tuple = ()
    extra: list = dataclasses.field(default_factory=list)


class Pair(typing.NamedTuple):
    a: float
    b: float = 2.0


class Shape(typing.TypedDict):
    a: float


class Tier(enum.Enum):
    GOLD = 1


T = typing.TypeVar("T")


class Base(abc.ABC, typing.Generic[T]):
    __slots__ = ("x",)
    LIMIT = 3.0

    @abc.abstractmethod
    def f(self): ...

    @property
    def p(self):
        return 1

    @functools.cached_property
    def q(self):
        return 2

    @functools.lru_cache(maxsize=None)
    def r(self):
        return 3

    @staticmethod
    def s():
        return 4

    class Nested:
        K = 1.0


def machinery(x: float) -> float:
    return x + len((Model, Terms, Pair, Shape, Tier, Base))


def test_class_machinery_is_not_hidden_state():
    assert hidden_state(machinery) == []
    fingerprint(machinery, strict=True)
