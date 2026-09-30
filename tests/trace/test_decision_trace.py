"""Structured decision tracing: equivalence, ordering, conservation, and the adapter seam."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from decider import branch, dag, flow, frame_step, loop, param, step
from decider.engine import Engine
from decider.engine.ir.origin import Origin
from decider.engine.trace import (Kind, PassThroughAdapter, StepTable, TraceEvent, TraceLossError, TraceSink,
                                  decode, pack, unpack)
from decider.engine.trace.envelope import SCHEMA_VERSION

MODES = ("interpreted", "stepped", "fused")


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def affordable(ratio: float) -> bool:
    return ratio >= 0.3


@step(output="term_cap")
def cap_private(term_cap: float, cap: float = param(54.0)) -> float:
    return min(term_cap, cap)


@step(output="term_cap")
def cap_public(term_cap: float, cap: float = param(60.0)) -> float:
    return min(term_cap, cap)


def is_private(sector_code: int) -> bool:
    return sector_code == 1


def term_cap(requested_term: float) -> float:
    return requested_term


@frame_step(reads=["client_id"], writes=["bureau_score"])
def join_bureau(df: pl.DataFrame) -> pl.DataFrame:
    return df.join(pl.DataFrame({"client_id": [1, 2], "bureau_score": [700, 650]}), on="client_id", how="left")


AFFORDABILITY = dag(disposable_income, ratio, affordable, name="affordability")
TERM = flow(term_cap, branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector"),
            name="term")
PIPELINE = (join_bureau | AFFORDABILITY | TERM).emit("term_cap@*")

FRAME = pl.DataFrame({
    "client_id": [1, 2, 3],
    "net_income": [9200.0, 4100.0, 15000.0],
    "expenses": [3100.0, 3700.0, 6000.0],
    "instalment": [1200.0, 800.0, 5000.0],
    "requested_term": [72.0, 50.0, 84.0],
    "sector_code": [1, 2, 1],
})


def step_events(exe):
    sink = TraceSink()
    exe.run(FRAME, trace=sink)
    return sink


def step_shapes(events):
    return sorted((e.kind.value, e.step, e.arm, e.iteration, e.record) for e in events if e.kind is Kind.STEP)


def test_tracing_does_not_change_the_output():
    expected = Engine().bind(PIPELINE, mode="interpreted").run(FRAME)
    for mode in MODES:
        exe = Engine().bind(PIPELINE, mode=mode)
        sink = TraceSink()
        out = exe.run(FRAME, trace=sink)
        assert out.equals(expected)


def test_tracing_is_off_by_default():
    # No sink passed: `run` captures nothing and returns the same result.
    exe = Engine().bind(PIPELINE, mode="fused")
    assert exe.run(FRAME).equals(Engine().bind(PIPELINE, mode="interpreted").run(FRAME))


def test_every_step_event_resolves_to_a_durable_origin():
    exe = Engine().bind(PIPELINE, mode="fused")
    sink = step_events(exe)
    ref = {c.id + 1: c.node.origin for c in exe.plan.calls}
    steps = [e for e in sink.events() if e.kind is Kind.STEP]
    assert steps
    for e in steps:
        assert e.step in ref
        assert e.origin == ref[e.step]
        assert e.origin.source


def test_step_evidence_is_equivalent_across_modes():
    shapes = {mode: step_shapes(step_events(Engine().bind(PIPELINE, mode=mode)).events()) for mode in MODES}
    assert shapes["stepped"] == shapes["interpreted"]
    assert shapes["fused"] == shapes["interpreted"]


def test_per_record_order_within_a_stream():
    # Within one record, STEP events are in step order (by `step`), whatever
    # the cross-record order (which is left unspecified).
    exe = Engine().bind(PIPELINE, mode="fused")
    sink = step_events(exe)
    by_record: dict[int, list[int]] = {}
    for e in sink.events():
        if e.kind is Kind.STEP:
            by_record.setdefault(e.record, []).append(e.step)
    for steps in by_record.values():
        assert steps == sorted(steps)
        assert len(set(steps)) == len(steps)


def test_branch_arms_carry_the_taken_arm():
    exe = Engine().bind(PIPELINE, mode="interpreted")
    sink = step_events(exe)
    private = [e for e in sink.events() if e.kind is Kind.STEP and e.origin.path.endswith("cap_private")]
    public = [e for e in sink.events() if e.kind is Kind.STEP and e.origin.path.endswith("cap_public")]
    assert [e.record for e in private] == [0, 2]
    assert [e.record for e in public] == [1]
    assert all(e.arm == 0 for e in private)
    assert all(e.arm == 1 for e in public)


def test_frame_steps_emit_a_frame_event_per_row():
    exe = Engine().bind(PIPELINE, mode="fused")
    sink = step_events(exe)
    frames = [e for e in sink.events() if e.kind is Kind.FRAME]
    assert [e.record for e in frames] == [0, 1, 2]
    assert all(e.origin.path.endswith("join_bureau") for e in frames)


def test_score_captures_one_records_evidence():
    exe = Engine().bind(AFFORDABILITY, mode="fused")
    sink = TraceSink()
    exe.score({"net_income": 4100.0, "expenses": 3700.0, "instalment": 800.0}, trace=sink)
    steps = [e for e in sink.events() if e.kind is Kind.STEP]
    assert [e.origin.path for e in steps] == [
        "affordability/disposable_income", "affordability/ratio", "affordability/affordable",
    ]
    assert all(e.record == 0 for e in steps)


def test_conservation_raises_when_a_trace_point_is_dropped():
    sink = TraceSink()
    header = np.array([pack(Kind.STEP, 1)], dtype=np.int64)
    offsets = np.array([1], dtype=np.int64)
    table = StepTable((Origin("a", "m:1"),))
    with pytest.raises(TraceLossError):
        sink.drain_kernel(header, offsets, table, 1, expected=3)


def test_conservation_passes_when_counts_match():
    sink = TraceSink()
    header = np.array([pack(Kind.STEP, 1), pack(Kind.STEP, 2)], dtype=np.int64)
    offsets = np.array([1, 2], dtype=np.int64)
    table = StepTable((Origin("a", "m:1"), Origin("b", "m:2")))
    sink.drain_kernel(header, offsets, table, 2, expected=2)
    assert [e.origin.path for e in sink.events()] == ["a", "b"]


def test_envelope_round_trips_and_decodes_to_origin():
    value = pack(Kind.STEP, 3, 1, 2)
    assert unpack(value) == (Kind.STEP.value, 3, 1, 2)
    table = StepTable((Origin("a", "m:1"), Origin("b", "m:2"), Origin("c", "m:3")))
    events = decode(np.array([value]), table, np.array([7]))
    assert events == [TraceEvent(SCHEMA_VERSION, Kind.STEP, 3, Origin("c", "m:3"), 0, 2, 7)]


def test_the_default_adapter_collects_everything():
    sink = TraceSink()
    sink.emit(Kind.STEP, Origin("a", "m:1"), step=1, record=0)
    assert len(sink.events()) == 1
    assert sink.status() == {"delivered": 1, "dropped": 0, "failed": 0, "errors": 0}


def test_a_custom_adapter_receives_events_and_failure_is_reported():
    adapter = PassThroughAdapter()
    sink = TraceSink(adapter)
    sink.emit(Kind.STEP, Origin("a", "m:1"), step=1, record=0)
    assert len(adapter.events) == 1

    def boom(events):
        raise RuntimeError("nope")

    sink2 = TraceSink(boom)
    sink2.emit(Kind.STEP, Origin("a", "m:1"), step=1, record=0)
    assert sink2.status() == {"delivered": 0, "dropped": 0, "failed": 1, "errors": 1}


def test_batch_capture_is_bounded_and_loss_is_reported():
    exe = Engine().bind(AFFORDABILITY, mode="fused")
    sink = TraceSink(max_events=5)
    exe.run(FRAME, trace=sink)
    assert sink.status()["delivered"] == 5
    assert sink.status()["dropped"] > 0
