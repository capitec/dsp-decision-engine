"""A kernel numba refuses only at launch runs in Python and is reported like any fallback: fused warns, stepped lists."""
import warnings
from typing import TypedDict

import polars as pl
import pytest
from numba.core.errors import TypingError

from decikit import allow_fallback, flow
from decikit._engine import Engine
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


FRAME = pl.DataFrame({"x": [1.0, 4.0]})


def caught_fallbacks(run) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run()
    return [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]


@pytest.mark.parametrize("fn", [offer, loose, offers])
def test_a_kernel_returning_a_dict_splits_when_built_before_any_row_runs(fn):
    exe = Engine().bind(flow(fn, name="p"), mode="fused")
    with pytest.warns(FallbackWarning, match=f"p/{fn.__name__} runs in Python, row by row: split at launch: "):
        reason = exe.fallbacks()[f"p/{fn.__name__}"]
    assert reason.startswith("split at launch: NumbaNotImplementedError: ")
    assert "Cannot cast" in reason


def test_fused_warns_once_about_a_kernel_that_splits_at_launch():
    exe = Engine().bind(flow(offer, name="p"), mode="fused")
    found = caught_fallbacks(lambda: (exe.run(FRAME), exe.run(FRAME), exe.score({"x": 1.0})))
    assert [m.split(":")[0] for m in found] == ["p/offer runs in Python, row by row"]


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_strict_compile_refuses_a_kernel_that_splits_at_launch_on_every_call(mode):
    exe = Engine(strict_compile=True).bind(flow(offer, name="p"), mode=mode)
    for _ in range(2):
        with pytest.raises(ValueError, match="(?s)p/offer runs in Python, row by row: split at launch: .*@allow_fallback"):
            exe.run(FRAME)


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_allow_fallback_accepts_a_kernel_that_splits_at_launch_silently(mode):
    exe = Engine(strict_compile=True).bind(flow(accepted, name="p"), mode=mode)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert exe.score({"x": 1.0})["accepted"] == {"v": 2.0}
    assert exe.fallbacks()["p/accepted"].startswith("@allow_fallback: split at launch: ")


def test_strict_compile_refuses_a_split_kernel_unless_every_step_in_it_is_allow_fallback():
    exe = Engine(strict_compile=True).bind(flow(half, accepted, name="p"), mode="fused")
    with pytest.raises(ValueError, match="p/half runs in Python, row by row: split at launch: "):
        exe.fallbacks()


def test_stepped_lists_a_kernel_that_splits_at_launch_without_a_warning():
    exe = Engine().bind(flow(offer, name="p"), mode="stepped")
    assert caught_fallbacks(lambda: exe.run(FRAME)) == []
    assert exe.fallbacks()["p/offer"].startswith("split at launch: NumbaNotImplementedError: ")


def test_interpreted_reports_nothing():
    exe = Engine(strict_compile=True).bind(flow(offer, name="p"), mode="interpreted")
    assert caught_fallbacks(lambda: exe.run(FRAME)) == []
    assert exe.fallbacks() == {}


def test_every_step_of_a_split_kernel_runs_in_python_with_the_answers_of_interpreted():
    pipeline = flow(half, offer, name="p")
    with pytest.warns(FallbackWarning):
        out = assert_equivalent(pipeline, FRAME)
    assert out["half"].to_list() == [0.5, 2.0] and out["offer"].to_list() == [{"v": 2.0}, {"v": 8.0}]
    exe = Engine().bind(pipeline, mode="fused")
    with pytest.warns(FallbackWarning):
        assert set(exe.fallbacks()) == {"p/half", "p/offer"}


def refused(*args):
    raise TypingError("a type the zero-row launch did not see")


@pytest.mark.parametrize("mode, warned", [("stepped", 0), ("fused", 1)])
def test_a_kernel_that_splits_after_the_build_is_reported_once(mode, warned):
    exe = Engine().bind(flow(half, name="p"), mode=mode)
    assert exe.fallbacks() == {}
    exe.runner.units[exe.plan.calls[0].id].fn = refused
    found = caught_fallbacks(lambda: (exe.run(FRAME), exe.run(FRAME)))
    assert len(found) == warned
    assert exe.run(FRAME)["half"].to_list() == [0.5, 2.0]
    reason = exe.fallbacks()["p/half"]
    assert reason.startswith("split at launch: TypingError: ") and "a type the zero-row launch did not see" in reason


def test_strict_compile_refuses_a_kernel_that_splits_after_the_build_on_every_call():
    exe = Engine(strict_compile=True).bind(flow(half, name="p"), mode="fused")
    assert exe.fallbacks() == {}
    exe.runner.units[exe.plan.calls[0].id].fn = refused
    for _ in range(2):
        with pytest.raises(ValueError, match="p/half runs in Python, row by row: split at launch: TypingError"):
            exe.run(FRAME)
    assert exe.fallbacks() == {}
