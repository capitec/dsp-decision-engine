"""A step returning what no kernel column stores falls back at build; a kernel numba refuses at launch is reported then.

Fused warns about both, stepped only lists them, interpreted reports nothing, and strict refuses both.
"""
import warnings
from typing import NewType, TypedDict

import numpy as np
import polars as pl
import pytest
from numba.core.errors import TypingError

from decikit import allow_fallback, flow
from decikit._engine import Engine
from decikit._engine.compile import Fallback, Kernel
from decikit.exceptions import FallbackWarning
from decikit.lab.testing import assert_equivalent


class Offer(TypedDict):
    v: float


def offer(x: float) -> Offer:
    return {"v": x * 2}


def loose(x: float) -> dict:
    return {"v": x}


def offers(x: float) -> list[Offer]:
    return [{"v": x}, {"v": x + 1}]


@allow_fallback
def accepted(x: float) -> Offer:
    return {"v": x * 2}


def half(x: float) -> float:
    return x / 2


@allow_fallback
def halved(x: float) -> float:
    return x / 2


FRAME = pl.DataFrame({"x": [1.0, 4.0]})

Score = NewType("Score", float)


def as_numpy(x: float) -> np.float64:
    return x * 2


def as_score(x: float) -> Score:
    return x * 2


def as_either(x: float) -> int | float:
    return x * 2


def as_array(x: float) -> np.ndarray:
    return np.array([x, x])


def caught_fallbacks(run) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run()
    return [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]


@pytest.mark.parametrize("fn, annotation", [(offer, Offer), (loose, dict), (offers, list[Offer]),
                                            (as_array, np.ndarray)])
def test_a_step_returning_a_container_falls_back_when_built_naming_its_output(fn, annotation):
    exe = Engine().bind(flow(fn, name="p"), mode="fused")
    reason = f"writes '{fn.__name__}' as {annotation}, which no kernel stores"
    assert caught_fallbacks(exe.fallbacks) == [f"p/{fn.__name__} runs in Python, row by row: {reason}"]
    assert exe.fallbacks() == {f"p/{fn.__name__}": reason}


def test_the_steps_beside_a_dict_step_stay_compiled_with_the_answers_of_interpreted():
    pipeline = flow(half, offer, name="p")
    with pytest.warns(FallbackWarning):
        out = assert_equivalent(pipeline, FRAME)
    assert out["half"].to_list() == [0.5, 2.0] and out["offer"].to_list() == [{"v": 2.0}, {"v": 8.0}]
    exe = Engine().bind(pipeline, mode="fused")
    assert caught_fallbacks(lambda: (exe.run(FRAME), exe.run(FRAME), exe.run({"x": 1.0}))) == [
        f"p/offer runs in Python, row by row: writes 'offer' as {Offer}, which no kernel stores"]
    assert [type(exe.runner.units[c.id]) for c in exe.plan.calls] == [Kernel, Fallback]


def test_a_step_returning_an_array_answers_as_interpreted_rather_than_its_first_element():
    pipeline = flow(as_array, name="p")
    want = Engine().bind(pipeline, mode="interpreted").run(FRAME)
    with pytest.warns(FallbackWarning):
        assert Engine().bind(pipeline, mode="fused").run(FRAME).equals(want)


@pytest.mark.parametrize("fn", [as_numpy, as_score, as_either])
def test_a_step_writing_a_number_under_another_annotation_stays_in_the_shared_kernel(fn):
    exe = Engine(strict_compile=True).bind(flow(half, fn, name="p"), mode="fused")
    assert exe.run(FRAME)[fn.__name__].to_list() == [2.0, 8.0]
    assert exe.fallbacks() == {}
    first, second = (exe.runner.units[c.id] for c in exe.plan.calls)
    assert isinstance(first, Kernel) and first is second


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_strict_compile_refuses_a_dict_step_on_every_call(mode):
    exe = Engine(strict_compile=True).bind(flow(offer, name="p"), mode=mode)
    for _ in range(2):
        with pytest.raises(ValueError, match="p/offer runs in Python, row by row: writes 'offer' .*@allow_fallback"):
            exe.run(FRAME)


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_allow_fallback_accepts_a_dict_step_silently_and_its_neighbour_stays_compiled(mode):
    exe = Engine(strict_compile=True).bind(flow(half, accepted, name="p"), mode=mode)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert exe.run({"x": 1.0})["accepted"] == {"v": 2.0}
    assert exe.fallbacks() == {"p/accepted": f"@allow_fallback: writes 'accepted' as {Offer}, which no kernel stores"}


def test_stepped_lists_a_dict_step_without_a_warning():
    exe = Engine().bind(flow(offer, name="p"), mode="stepped")
    assert caught_fallbacks(lambda: exe.run(FRAME)) == []
    assert set(exe.fallbacks()) == {"p/offer"}


def test_interpreted_reports_nothing():
    exe = Engine(strict_compile=True).bind(flow(offer, name="p"), mode="interpreted")
    assert caught_fallbacks(lambda: exe.run(FRAME)) == []
    assert exe.fallbacks() == {}


def refused(*args):
    raise TypingError("a type the build did not see")


def split_after_the_build(exe):
    assert exe.fallbacks() == {}
    exe.runner.units[exe.plan.calls[0].id].fn = refused


@pytest.mark.parametrize("mode, warned", [("stepped", 0), ("fused", 1)])
def test_a_kernel_that_splits_at_launch_is_reported_once(mode, warned):
    exe = Engine().bind(flow(half, name="p"), mode=mode)
    split_after_the_build(exe)
    assert len(caught_fallbacks(lambda: (exe.run(FRAME), exe.run(FRAME)))) == warned
    assert exe.run(FRAME)["half"].to_list() == [0.5, 2.0]
    reason = exe.fallbacks()["p/half"]
    assert reason.startswith("split at launch: TypingError: ") and "a type the build did not see" in reason


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_strict_compile_refuses_a_kernel_that_splits_at_launch_on_every_call(mode):
    exe = Engine(strict_compile=True).bind(flow(half, name="p"), mode=mode)
    split_after_the_build(exe)
    for _ in range(2):
        with pytest.raises(ValueError, match="p/half runs in Python, row by row: split at launch: TypingError"):
            exe.run(FRAME)
    assert exe.fallbacks() == {}


def test_allow_fallback_accepts_a_kernel_that_splits_at_launch_silently():
    exe = Engine(strict_compile=True).bind(flow(halved, name="p"), mode="fused")
    split_after_the_build(exe)
    assert caught_fallbacks(lambda: exe.run(FRAME)) == []
    assert exe.fallbacks()["p/halved"].startswith("@allow_fallback: split at launch: TypingError: ")
