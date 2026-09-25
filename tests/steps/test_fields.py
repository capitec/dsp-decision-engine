from typing import Annotated

import polars as pl

from decider import Duration, Engine, Money, Percent, flow, param
from decider.fields import metadata_of


def instalment(loan: Annotated[float, Money()], rate: Annotated[float, Percent()] = param(0.12)) -> Annotated[float, Money()]:
    return loan * rate / 12


def test_field_metadata_is_read_from_annotated_types():
    assert metadata_of(Annotated[float, Money(cents=True)]) == Money(cents=True)
    assert metadata_of(Annotated[int, "unrelated", Duration("days")]) == Duration("days")
    assert metadata_of(float) is None and metadata_of(Annotated[float, "unrelated"]) is None


def test_field_metadata_is_json_with_its_kind():
    assert Money().to_json() == {"kind": "money", "symbol": "R", "cents": False}
    assert Duration("days").to_json() == {"kind": "duration", "unit": "days"}


def test_annotated_steps_run_like_plain_ones_in_every_mode():
    p = flow(instalment, name="f")
    assert p.run(pl.DataFrame({"loan": [1200.0]}))["instalment"].to_list() == [12.0]
    for mode in ("fused", "stepped"):
        assert Engine().bind(p, mode=mode).score({"loan": 1200.0})["instalment"] == 12.0
    assert p.parameters().defaults() == {"f": {"instalment": {"rate": 0.12}}}
