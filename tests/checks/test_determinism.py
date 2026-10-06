"""Determinism checks: wall-clock reads, shared state, in-place changes to arguments."""
import random
import time
from datetime import datetime as dt
from time import time as now

from decider import checks, flow, step


def _found(pipeline, alias):
    return [f for f in checks.run(pipeline, [alias]).findings if f.check == alias]


@step(output="a")
def from_import(x: float) -> float:
    return x + now()


@step(output="b")
def aliased_class(x: float) -> float:
    return x + dt.now().second


@step(output="c")
def local_alias(x: float) -> float:
    import time as t

    return x + t.time()


@step(output="d")
def from_random(x: float) -> float:
    return x * random.random()


@step(output="e")
def clean(x: float) -> float:
    return x + time.strptime("2020", "%Y").tm_year


def test_wall_clock_resolves_from_imports_and_aliases():
    pipeline = flow(from_import, aliased_class, local_alias, from_random, name="f")
    calls = sorted(f.detail["call"] for f in _found(pipeline, "wall_clock"))
    assert calls == ["dt.now()", "now()", "random.random()", "t.time()"]


def test_wall_clock_is_quiet_on_other_time_functions():
    assert not _found(flow(clean, name="f"), "wall_clock")


def test_impure_flags_a_keyword_only_mutable_default_and_says_why():
    @step(output="y")
    def kw(x: float) -> float:
        def helper(*, seen=[]):  # noqa: B006
            return seen

        return x

    (f,) = _found(flow(kw, name="f"), "impure")
    assert f.detail["arg"] == "seen" and "concurrently" in f.message


def test_impure_message_explains_the_race_for_globals():
    @step(output="y")
    def leaky(x: float) -> float:
        global _STATE
        _STATE = x
        return x

    (f,) = _found(flow(leaky, name="f"), "impure")
    assert "Executable" in f.message


def test_mutates_input_reports_method_item_and_inplace_changes():
    @step(output="y")
    def bad(items: list, frame: dict) -> float:
        items.append(1)
        frame["k"] = 2
        frame.sort(inplace=True)
        return 0.0

    found = sorted(_found(flow(bad, name="f"), "mutates_input"), key=lambda f: f.line)
    assert [(f.code, f.detail["arg"], f.detail["how"]) for f in found] == [
        ("DCD003", "items", ".append()"), ("DCD003", "frame", "item assignment"),
        ("DCD003", "frame", ".sort(inplace=True)")]
    assert "concurrently" in found[0].message


def test_mutates_input_is_quiet_on_a_copy_or_a_local():
    @step(output="y")
    def good(items: list) -> float:
        items = list(items)
        items.append(1)
        local = []
        local.append(2)
        return float(len(items) + len(local))

    assert not _found(flow(good, name="f"), "mutates_input")
