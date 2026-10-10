"""`each` inside `each`, and `each` beside other steps in a `dag`: a child's names belong to its element."""
from __future__ import annotations

import pytest

from decikit import Engine, allow_fallback, dag, each, flow, step

MODES = ("interpreted", "fused")


def double(x: float) -> float:
    return x * 2


@allow_fallback
def total(inner: list[dict]) -> float:
    return sum(i["double"] for i in inner)


def entity(output: str):
    return flow(each("items", flow(step(double, output="double"), name="item"), name="items", output=output),
                step(total, output="total"), name="entity")


RECORD = {"entities": [{"items": [{"x": 1.0}, {"x": 2.0}]}]}


@pytest.mark.parametrize("mode", MODES)
def test_a_nested_each_writing_another_column_gathers_its_items(mode):
    p = flow(each("entities", entity("inner"), name="entities"), name="p")
    got = Engine().bind(p, mode=mode).score(RECORD)["entities"][0]
    assert got["inner"] == [{"x": 1.0, "double": 2.0}, {"x": 2.0, "double": 4.0}]
    assert got["total"] == 6.0


@pytest.mark.parametrize("mode", MODES)
def test_a_nested_each_overwriting_its_own_column_gathers_its_items(mode):
    inner = flow(each("items", flow(step(double, output="double"), name="item"), name="items"), name="entity")
    p = flow(each("entities", inner, name="entities"), name="p")
    got = Engine().bind(p, mode=mode).score(RECORD)["entities"][0]
    assert got["items"] == [{"x": 1.0, "double": 2.0}, {"x": 2.0, "double": 4.0}]


def rate() -> float:
    return 1.0


def reads_rate(rate: float) -> float:
    return rate + 1


def test_a_dag_does_not_count_an_eachs_child_names_as_its_own_writes():
    def g(xs: list[dict]) -> float:
        return float(len(xs))

    child = flow(step(rate, output="rate"), step(reads_rate, output="y"), name="item")
    p = dag(each("xs", child, name="xs"), step(rate, output="rate"), step(g, output="g"), name="p")
    out = Engine().bind(p.emit("xs", "rate", "g"), mode="interpreted").score({"xs": [{}, {}]})
    assert out["xs"] == [{"rate": 1.0, "y": 2.0}, {"rate": 1.0, "y": 2.0}]
    assert out["g"] == 2.0
