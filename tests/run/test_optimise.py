"""`optimise`: generate candidates, score each, keep the best per row."""
from __future__ import annotations

from typing import TypedDict

import polars as pl
import pytest

from decider import Columnar, Engine, flow, optimise, param, step
from decider.exceptions import WiringError
from decider.testing import assert_equivalent

MODES = ("interpreted", "stepped", "fused")


class Item(TypedDict):
    price: float
    weight: float


@step(output="count")
def bundles(items: Columnar[Item]) -> int:
    return (1 << len(items.price)) - 1


@step(output="bundle_weight")
def bundle_weight(index: int, items: Columnar[Item]) -> float:
    w = 0.0
    for j in range(len(items.weight)):
        if (index >> j) & 1:
            w += items.weight[j]
    return w


@step(output="score")
def total(index: int, items: Columnar[Item]) -> float:
    t = 0.0
    for j in range(len(items.price)):
        if (index >> j) & 1:
            t += items.price[j]
    return t


def heavy(bundle_weight: float, max_weight: float = param(30.0)) -> bool:
    return bundle_weight > max_weight


def pipeline(disqualify=None, score="score"):
    return flow(optimise(bundles, flow(bundle_weight, total, name="evaluate"),
                         score=score, disqualify=disqualify, max_candidates=1 << 8, name="best"),
                name="order")


FRAME = pl.DataFrame({"items": [
    [{"price": 2.0, "weight": 1.0}, {"price": 5.0, "weight": 2.0}, {"price": 3.0, "weight": 20.0}],
    [],
    [{"price": 4.0, "weight": 50.0}, {"price": 1.0, "weight": 1.0}],
]})


def test_keeps_the_highest_scoring_bundle_in_every_mode():
    out = assert_equivalent(pipeline(), FRAME)
    assert out["best_index"].to_list() == [7, -1, 3]
    assert out["best_score"].to_list() == [10.0, -1e300, 5.0]
    assert out["evaluated"].to_list() == [7, 0, 3]
    assert out["disqualified"].to_list() == [0, 0, 0]


def test_a_disqualified_bundle_is_counted_and_excluded():
    exe = Engine().bind(pipeline(disqualify=heavy), mode="fused")
    out = exe.run(FRAME)
    assert out["best_index"].to_list() == [7, -1, 2]
    assert out["disqualified"].to_list() == [0, 0, 2]


def test_an_empty_list_reads_as_no_candidates():
    exe = Engine().bind(pipeline(), mode="fused")
    got = exe.score({"items": []})
    assert got["best_index"] == -1 and got["best_score"] == -1e300
    assert got["evaluated"] == 0 and got["disqualified"] == 0


def test_a_tie_keeps_the_earlier_candidate():
    frame = pl.DataFrame({"items": [[{"price": 2.0, "weight": 1.0}, {"price": 2.0, "weight": 1.0},
                                     {"price": 0.0, "weight": 1.0}]]})
    out = assert_equivalent(pipeline(), frame)
    assert out["best_index"].to_list() == [3]  # bundles {0,1} and {0,1,2} both score 4; earlier wins


def test_the_score_output_is_renamable():
    evaluate = flow(bundle_weight, total.relabel(writes={"score": "margin"}), name="evaluate")
    p = flow(optimise(bundles, evaluate, score="margin", max_candidates=1 << 8, name="best"), name="order")
    out = Engine().bind(p, mode="fused").run(FRAME)
    assert out["best_score"].to_list() == [10.0, -1e300, 5.0]


def test_the_childrens_params_are_tunable_through_the_document():
    exe = Engine().bind(pipeline(disqualify=heavy), mode="fused")
    assert exe.run(FRAME)["disqualified"].to_list() == [0, 0, 2]
    params = {"order": {"best": {"search": {"body": {"heavy": {"max_weight": 1.0}}}}}}
    out = exe.run(FRAME, params=params)
    # max_weight 1 disqualifies the weight-2 bundle and the weight-20 one, etc.
    assert out["disqualified"].to_list() == [6, 0, 2]


def test_a_missing_score_output_is_rejected():
    with pytest.raises(WiringError):
        optimise(bundles, flow(total, name="evaluate"), score="margin", max_candidates=8, name="best")
