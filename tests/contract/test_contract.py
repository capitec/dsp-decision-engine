from __future__ import annotations

import warnings

import pytest

from decider import branch, dag, flow, param, step
from decider.contract import (CONTRACT_VERSION, EdgeRef, FlowDescription, FlowRef, RecordRef, StepRef, ValueSlotRef,
                              capabilities, check_version, describe, json_schema, mcp_schema, resolve_step, typescript)
from decider.exceptions import DeciderError


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def affordable(ratio: float, min_ratio: float = param(0.3)) -> bool:
    return ratio >= min_ratio


@step(output="term_cap")
def cap_by_income(term_cap: float, cap: float = param(48.0)) -> float:
    return min(term_cap, cap)


def is_private(sector_code: int) -> bool:
    return sector_code == 1


@step(output="term_cap")
def cap_private(term_cap: float, cap: float = param(54.0)) -> float:
    return min(term_cap, cap)


@step(output="term_cap")
def cap_public(term_cap: float, cap: float = param(60.0)) -> float:
    return min(term_cap, cap)


PIPELINE = flow(
    dag(ratio, affordable, name="affordability"),
    cap_by_income,
    branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector"),
    name="term",
)


def test_describe_names_nodes_edges_and_slots():
    desc = describe(PIPELINE)
    assert isinstance(desc, FlowDescription)
    assert desc.contract_version == CONTRACT_VERSION
    paths = {n.path for n in desc.nodes}
    assert "term/affordability/ratio" in paths
    assert "term/cap_by_income" in paths
    control = {(e.from_path, e.to_path) for e in desc.edges if e.kind == "control"}
    assert ("term/by_sector", "term/by_sector/cap_private") in control
    slots = {v.spec for v in desc.value_slots}
    assert "term_cap@term/cap_by_income" in slots


def test_data_edges_carry_the_value_the_target_reads():
    desc = describe(PIPELINE)
    data = {e: e.values for e in desc.edges if e.kind == "data"}
    assert any(e.from_path == "term/affordability/ratio" and e.to_path == "term/affordability/affordable"
               and e.values == ("ratio",) for e in data)


def test_value_slot_spec_matches_state_syntax():
    slot = ValueSlotRef(name="term_cap", path="term/cap_by_income")
    assert slot.spec == "term_cap@term/cap_by_income"
    assert ValueSlotRef(name="sector_code").spec == "sector_code"


def test_durable_record_ref_requires_a_key():
    ref = RecordRef(dataset="loans.parquet", key={"client_id": "C-1"})
    assert ref.durable
    assert not RecordRef(dataset="loans.parquet", ordinal=7).durable


def test_resolve_step_matches_by_path_when_no_durable_id():
    desc = describe(PIPELINE)
    by_path = resolve_step(desc, StepRef(path="term/cap_by_income"))
    assert by_path.path == "term/cap_by_income"
    assert by_path.node_type == "call"


def test_unresolved_durable_reference_warns_and_keeps_metadata():
    desc = describe(PIPELINE)
    ref = StepRef(flow_id="f1", step_id="gone", path="term/cap_by_income", source="credit.rules:cap_by_income")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = resolve_step(desc, ref)
    assert out is ref
    assert any("gone" in str(w.message) for w in caught)


def test_check_version_rejects_a_newer_contract():
    check_version(None)
    check_version(CONTRACT_VERSION)
    with pytest.raises(DeciderError):
        check_version(str(int(CONTRACT_VERSION) + 1))


def test_json_schema_is_versioned_and_self_contained():
    schema = json_schema()
    assert schema["contract_version"] == CONTRACT_VERSION
    assert "StepRef" in schema["$defs"]
    assert "FlowDescription" in schema["$defs"] or schema["title"] == "decider static-flow contract"
    assert schema == mcp_schema()


def test_typescript_generates_every_reference_type():
    ts = typescript()
    for name in ("FlowRef", "StepRef", "EdgeRef", "ValueSlotRef", "RecordRef", "FlowDescription"):
        assert f"export interface {name} " in ts


def test_capabilities_report_supported_modes():
    caps = capabilities()
    assert caps.contract_version == CONTRACT_VERSION
    assert "interpreted" in caps.modes
