"""The decision end to end: the sample, order independence, replay, batch and the evidence it records."""
import copy
import random
from datetime import date

import polars as pl
import pytest
from conftest import entity, event, parse_dates

from business_nested.pipeline import DECISION_OUTPUTS, build


def test_the_sample_is_declined_and_says_which_entity_and_which_event(assess):
    out = assess()
    assert out["decision_id"] == "biz-2026-0900123"
    assert (out["outcome_code"], out["primary_reason_code"]) == (4, 5402)
    assert (out["attributing_entity_id"], out["attributing_event_ids"], out["attributing_rule_id"]) == (1, [41], "AE-R-01")
    ms_dlamini = out["assessed_entities"][0]
    assert ms_dlamini["entity_key"] == "PERSON-MDLAMINI" and ms_dlamini["effective_ownership_pct"] == 32.0
    assert ms_dlamini["criticality_class"] == 1  # a required surety
    event_41 = ms_dlamini["adverse_events"][0]
    assert (event_41["event_id"], event_41["event_severity_code"], event_41["material_threshold_base"]) == (41, 4, 10_000.0)
    assert event_41["disqualifying_threshold"] == 50_000.0


def test_the_whole_nested_assessment_is_in_the_result(assess):
    entities = assess()["assessed_entities"]
    assert [e["entity_id"] for e in entities] == [1, 2, 3]
    assert [e["path_count"] for e in entities] == [1, 2, 1]
    assert [len(e["adverse_events"]) for e in entities] == [2, 0, 3]
    assert [e["adverse_verdict_code"] for e in entities] == [4, 1, 3]
    assert [e["adverse_binding_rule"] for e in entities] == ["AE-R-01", "AE-R-11", "AE-R-02"]
    assert entities[2]["adverse_event_ids"] == [51, 52, 53]  # three minor events: all three are named


def test_without_the_judgment_the_same_business_is_approved_and_priced(assess, sample):
    sample["entities"][0]["adverse_events"].pop(0)
    out = assess(sample["entities"])
    assert (out["outcome_code"], out["risk_grade"], out["offered_amount"]) == (1, 7, 3_500_000.0)
    assert out["instalment"] == 106_202.67


def shuffled(record, rng):
    record = copy.deepcopy(record)
    rng.shuffle(record["entities"])
    for e in record["entities"]:
        rng.shuffle(e["adverse_events"])
    return record


def by_entity(out, field):
    return {e["entity_id"]: e[field] for e in out["assessed_entities"]}


def test_outcome_and_attribution_do_not_depend_on_the_order_of_entities_and_events(executable, params, sample):
    baseline = executable.score(parse_dates(sample), params)
    rng = random.Random(7)
    for _ in range(10):
        out = executable.score(parse_dates(shuffled(sample, rng)), params)
        assert {name: out[name] for name in DECISION_OUTPUTS} == {name: baseline[name] for name in DECISION_OUTPUTS}
        assert by_entity(out, "adverse_event_ids") == by_entity(baseline, "adverse_event_ids")


def test_order_independence_holds_for_a_structure_with_two_paths_and_many_events(executable, params, sample):
    many = [event(i, 800.0 * i, event_date=date(2026, 1, i)) for i in range(1, 16)]
    sample["entities"] = [entity(1, 5, 30.0, many), entity(2, 1, 0.0, is_controlling=True, entity_key="K"),
                          entity(2, 5, 50.0, many[:3], entity_key="K"), entity(3, 5, 20.0)]
    baseline = executable.score(parse_dates(sample), params)
    rng = random.Random(11)
    for _ in range(10):
        out = executable.score(parse_dates(shuffled(sample, rng)), params)
        assert {name: out[name] for name in DECISION_OUTPUTS} == {name: baseline[name] for name in DECISION_OUTPUTS}


def test_replaying_a_decision_reproduces_it_exactly(executable, params, sample):
    record = parse_dates(sample)
    assert executable.score(record, params) == executable.score(copy.deepcopy(record), params)


def test_a_batch_agrees_with_one_record_at_a_time(executable, params, sample):
    declined, approved = parse_dates(sample), parse_dates(sample)
    approved["entities"][0]["adverse_events"].pop(0)
    batch = executable.run(pl.DataFrame([declined, approved]), params)
    for row, record in enumerate((declined, approved)):
        single = executable.score(record, params)
        for name in ("outcome_code", "risk_grade", "offered_amount", "instalment", "primary_reason_code",
                     "attributing_entity_id", "attributing_event_ids", "decline_reason_codes"):
            assert batch[name].to_list()[row] == single[name], name
    assert batch["outcome_code"].to_list() == [4, 1]


def test_every_table_read_is_recorded_with_its_version_and_cell(assess):
    out = assess()
    assert out["risk_grade_cell_id"] == "grade_ladder@biz-rg-2026.09#50|7"
    assert out["rate_card_cell_id"] == "business_rate_card@biz-rate-2026.09#7|2"
    assert out["appetite_cell_id"] == "business_appetite@biz-rate-2026.09#7"
    assert out["adjustment_set_id"] == "AS-05-BIZ-2026.09"
    first = out["assessed_entities"][0]
    assert first["adverse_events"][0]["threshold_cell_id"] == "event_thresholds@ev-thresh-2026.09#1|1"
    assert first["entity_grade_cell_id"].startswith("grade_ladder@bus-rg-2026.09#4|")
    assert first["calibration_cell_id"] == "calibration@cal-2026.09#4"


def test_the_decision_date_governs_which_overlays_apply(assess):
    in_force, lapsed = assess(), assess(decision_date=date(2027, 1, 10))
    assert in_force["assessed_entities"][0]["adverse_events"][1]["threshold_overlay_ids"] == ["ADJ-05-014"]
    assert lapsed["assessed_entities"][0]["adverse_events"][1]["threshold_overlay_ids"] == []
    assert lapsed["adjustments_applied"] == []


def test_the_shipped_params_document_is_the_pipelines_defaults(params):
    assert params == build().parameters().defaults()
