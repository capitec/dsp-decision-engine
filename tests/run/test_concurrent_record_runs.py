"""Concurrent calls on one bound `Executable` give correct, uncorrupted results."""
import threading

import polars as pl

from decikit import Engine, flow, missing_as, param


def heavy(weight: float = missing_as(0.0), heavy_kg: float = param(20.0)) -> bool:
    return weight > heavy_kg


def pipeline():
    from decikit import each

    return flow(each("items", flow(heavy, name="item"), name="items"), name="order")


def test_concurrent_record_runs_are_correct():
    exe = Engine().bind(pipeline(), mode="fused")
    errors = []

    def score():
        for _ in range(200):
            out = exe.run({"items": [{"weight": 5.0}, {"weight": 25.0}]})
            if out["items"] != [{"weight": 5.0, "heavy": False}, {"weight": 25.0, "heavy": True}]:
                errors.append(out)

    threads = [threading.Thread(target=score) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
