"""Binding a pipeline warns once per hidden object a scalar step reads, whatever the mode and `@allow_fallback`."""
import warnings

import polars as pl
import pytest

from decider import Engine, HiddenStateWarning, allow_fallback, flow, step
from decider._engine.run.hidden import warn_hidden_state
from decider.exceptions import EngineError


class Limits:
    cap = 100.0


LIMITS = Limits()


def capped(x: float) -> float:
    return min(x, LIMITS.cap)


@allow_fallback
def capped_declared(x: float) -> float:
    return min(x, LIMITS.cap)


def plain(x: float) -> float:
    return x * 2.0


def bind(*steps, mode="interpreted"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exe = Engine().bind(flow(*steps, name="p"), mode=mode)
    return exe, [w for w in caught if issubclass(w.category, HiddenStateWarning)]


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_binding_a_step_that_reads_a_global_object_warns_in_every_mode(mode):
    _, caught = bind(capped, mode=mode)
    [message] = [str(w.message) for w in caught]
    assert message.startswith("p/capped reads Limits `LIMITS` (module global in ")
    assert "decider.lab.diff cannot compare an object" in message
    assert "See `decider guide`, \"Shipping a step type in a package\"." in message


def test_allow_fallback_does_not_silence_the_warning():
    _, caught = bind(capped_declared)
    assert len(caught) == 1


def test_a_clean_step_does_not_warn_and_running_does_not_warn_again():
    exe, caught = bind(plain, capped)
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", HiddenStateWarning)
        assert exe.run(pl.DataFrame({"x": [1.0, 500.0]}))["capped"].to_list() == [1.0, 100.0]
        assert exe.score({"x": 3.0})["capped"] == 3.0


def test_each_hidden_variable_of_a_step_is_one_warning():
    other = Limits()

    def two(x: float) -> float:
        return min(x, LIMITS.cap, other.cap)

    _, caught = bind(two)
    assert sorted(str(w.message).split("`")[1] for w in caught) == ["LIMITS", "other"]


def test_strict_turns_the_warning_into_an_error():
    exe, _ = bind(capped)
    with pytest.raises(EngineError, match="p/capped reads Limits `LIMITS`"):
        warn_hidden_state(exe.plan, strict=True)


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_binding_with_strict_compile_raises_on_a_hidden_object(mode):
    with pytest.raises(EngineError, match="p/capped reads Limits `LIMITS`"):
        Engine(strict_compile=True).bind(flow(capped, name="p"), mode=mode)


def test_binding_with_strict_compile_accepts_a_clean_pipeline():
    Engine(strict_compile=True).bind(flow(plain, name="p"), mode="interpreted")
