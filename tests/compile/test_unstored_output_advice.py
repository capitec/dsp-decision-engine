"""A step that runs in Python because no kernel stores what it returns is told the declared form that compiles."""
import warnings
from datetime import date
from typing import TypedDict

import numpy as np
import polars as pl
import pytest

from decikit import FallbackWarning, FusedEngine, InterpretedEngine, flow, step
from decikit.ops import Columnar, Struct


class Offer(TypedDict):
    rate: float
    limit: float


class Amount(TypedDict):
    v: float


class Banded(TypedDict):
    band: str
    limit: float


def offer(x: float) -> dict:
    return {"rate": x, "limit": x * 2}


def typed_offer(x: float) -> Offer:
    return {"rate": x, "limit": x * 2}


def pair(x: float) -> tuple[float, float]:
    return x, x * 2


def amounts(x: float) -> list[float]:
    return [x, x * 2]


def offers(x: float) -> list[Offer]:
    return [{"rate": x, "limit": x * 2}, {"rate": x + 1, "limit": x * 3}]


def as_array(x: float) -> np.ndarray:
    return np.array([x, x * 2])


def tags(x: float) -> set:
    return {x}


def banded(x: float) -> Banded:
    return {"band": "A", "limit": x}


def names(x: float) -> list[str]:
    return ["a", "b"]


def due(x: float) -> date | None:
    return None  # a body numba compiles, so only the declared type decides


ONE_RECORD = ("to compile it, declare -> Struct[{item}] and return a tuple of its fields (one record), "
              "-> Columnar[{item}] and return a list of such tuples (many rows), or return a tuple under "
              "@step(outputs=(...)) (one output per value)")
MANY_ROWS = ("to compile it, declare -> Columnar[{item}] and return a list of tuples of its fields "
             "(a one-field {item} for a list of numbers)")
NO_FORM = "no compiled step returns one; accept it with @allow_fallback"

ADVICE = [
    (offer, "dict", ONE_RECORD.format(item="Item") + ", Item a TypedDict of the dict's keys"),
    (typed_offer, "Offer", ONE_RECORD.format(item="Offer")),
    (pair, "tuple[float, float]", "to compile it, return the tuple under @step(outputs=(...)) (one output per "
                                  "element), or declare -> Struct[Item] (one record)"),
    (amounts, "list[float]", MANY_ROWS.format(item="Item")),
    (offers, "list[Offer]", MANY_ROWS.format(item="Offer")),
    (as_array, "ndarray", MANY_ROWS.format(item="Item")),
    (tags, "set", NO_FORM),
    (due, "date", NO_FORM),
    (banded, "Banded", NO_FORM),
    (names, "list[str]", NO_FORM),
]


@pytest.mark.parametrize("fn, shown, advice", ADVICE, ids=[fn.__name__ for fn, _, _ in ADVICE])
def test_the_fallback_warning_names_the_step_and_the_form_that_compiles(fn, shown, advice):
    exe = FusedEngine().bind(flow(fn, name="p"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reason = exe.fallbacks()[f"p/{fn.__name__}"]
    assert reason == f"writes '{fn.__name__}' as {shown}, which no kernel stores; {advice}"
    assert [str(w.message) for w in caught if w.category is FallbackWarning] == [
        f"p/{fn.__name__} runs in Python, row by row: {reason}"]


def test_a_body_numba_refuses_keeps_its_reason_after_the_advice():
    def opened(x: int) -> date:
        return date.fromordinal(x)

    reason = FusedEngine().bind(flow(opened, name="p")).fallbacks()["p/opened"]
    assert reason.startswith(f"writes 'opened' as date, which no kernel stores; {NO_FORM}; numba also refused "
                             "the body: TypingError")


@step(output="offer")
def offer_struct(x: float) -> Struct[Offer]:
    return x, x * 2


@step(output="offers")
def offers_columnar(x: float) -> Columnar[Offer]:
    return [(x, x * 2), (x + 1, x * 3)]


@step(output="amounts")
def amounts_columnar(x: float) -> Columnar[Amount]:
    return [(x,), (x * 2,)]


@step(outputs=("rate", "limit"))
def offer_outputs(x: float) -> tuple[float, float]:
    return x, x * 2


FRAME = pl.DataFrame({"x": [1.0, 4.0]})


def answers(fn) -> list:
    exe = FusedEngine(strict_compile=True).bind(flow(fn, name="p"))
    out = exe.run(FRAME)
    assert exe.fallbacks() == {}
    assert out.equals(InterpretedEngine().bind(flow(fn, name="p")).run(FRAME))
    return out.drop("x").to_dicts()


def python_answers(fn) -> list:
    return [{fn.__name__: v} for v in InterpretedEngine().bind(flow(fn, name="p")).run(FRAME)[fn.__name__].to_list()]


def test_struct_compiles_and_answers_as_the_dict_it_replaces():
    assert answers(offer_struct) == [{"offer": d["offer"]} for d in python_answers(offer)]


def test_outputs_compiles_and_answers_as_the_dict_it_replaces():
    assert answers(offer_outputs) == [d["offer"] for d in python_answers(offer)]


def test_columnar_compiles_and_answers_as_the_typed_list_it_replaces():
    assert answers(offers_columnar) == [{"offers": d["offers"]} for d in python_answers(offers)]


def test_a_one_field_columnar_compiles_and_holds_the_list_of_numbers():
    assert answers(amounts_columnar) == [{"amounts": [{"v": v} for v in d["amounts"]]} for d in python_answers(amounts)]
