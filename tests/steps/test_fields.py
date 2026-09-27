from typing import Annotated, Optional

import polars as pl

from decider import Duration, Engine, Money, Percent, flow, param
from decider.engine.ir.decls import NullPolicy, base_annotation
from decider.engine.params.harvest import harvest
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
        exe = Engine().bind(p, mode=mode)
        assert exe.score({"loan": 1200.0})["instalment"] == 12.0
        assert exe.fallbacks() == {}          # the kernel reads the plain type, not one call per row
    assert p.parameters().defaults() == {"f": {"instalment": {"rate": 0.12}}}


def doubled(loan: Annotated[float | None, {"unit": "kg"}]) -> Annotated[float, Money()]:
    return 0.0 if loan is None else loan * 2


def test_a_declaration_keeps_its_metadata_and_the_engine_reads_through_it():
    # The flow debugger formats values by the metadata on the decls, so the decls keep it;
    # type logic sees the plain type, and metadata it can't hash never reaches a cache key.
    inputs, _, outputs = harvest(doubled)
    assert inputs[0].annotation == Annotated[float | None, {"unit": "kg"}]
    assert outputs[0].annotation == Annotated[float, Money()]
    assert metadata_of(harvest(instalment)[0][0].annotation) == Money()
    assert metadata_of(harvest(instalment)[2][0].annotation) == Money()

    assert (base_annotation(inputs[0].annotation), inputs[0].null_policy) == (float, NullPolicy.OPTIONAL)
    assert base_annotation(outputs[0].annotation) is float
    p = flow(doubled, name="f")
    df = pl.DataFrame({"loan": [1200.0, None]})
    for mode in ("interpreted", "stepped", "fused"):
        exe = Engine().bind(p, mode=mode)
        assert exe.run(df)["doubled"].to_list() == [2400.0, 0.0]
        assert exe.score({"loan": 1200.0})["doubled"] == 2400.0
        assert exe.fallbacks() == {}


def optional_outside(loan: Optional[Annotated[float, Money()]],
                     cap: Optional[Annotated[float, {"unit": "kg"}]] = param(9.0)) -> float:
    return 0.0 if loan is None else min(loan, cap)


def test_an_optional_around_annotated_is_canonicalised_so_both_readers_look_in_one_place():
    inputs, params, _ = harvest(optional_outside)
    assert inputs[0].annotation == Annotated[float | None, Money()]
    assert (metadata_of(inputs[0].annotation), inputs[0].null_policy) == (Money(), NullPolicy.OPTIONAL)
    # A param keeps the spelling it was given: pydantic reads its metadata, and a validator
    # inside `| None` checks something different from one outside it.
    assert params[0].annotation == Optional[Annotated[float, {"unit": "kg"}]]
    df = pl.DataFrame({"loan": [12.0, None]})
    for mode in ("interpreted", "stepped", "fused"):
        assert Engine().bind(flow(optional_outside, name="f"), mode=mode).run(df)["optional_outside"] \
            .to_list() == [9.0, 0.0]
