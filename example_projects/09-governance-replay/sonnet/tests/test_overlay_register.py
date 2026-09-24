"""The overlay register, ageing and stack-off run (spec 09 §5.14)."""
import json
from datetime import date

from governance import evidence_store, flows, overlay_register


def test_combined_register_covers_every_flow_s_overlays():
    rows = overlay_register.combined_register(("01", "03", "05"))
    flow_codes = {r.flow_code for r in rows}
    assert flow_codes == {"01", "03", "05"}
    ids = {r.adjustment_id for r in rows}
    assert "ADJ-2026-001" in ids  # 01's sensitivity dial
    assert "ADJ-05-ENT-001" in ids  # 05's entity-level overlay


def test_ageing_report_finds_overlays_due_for_review_far_enough_in_the_future():
    report = overlay_register.ageing_report(("01", "03", "05"), date(2028, 1, 1))
    assert report["count"] > 0
    assert all(row["age_days"] >= 0 for row in report["overlays_due_for_review"])


def test_ageing_report_is_empty_before_any_overlay_s_review_date():
    report = overlay_register.ageing_report(("01", "03", "05"), date(2025, 1, 1))
    assert report["count"] == 0


def test_stack_off_run_uses_the_same_implementation_through_one_param_flip():
    """§5.14.3: one implementation, not a second one -- proven by getting a real
    (possibly unchanged) counterfactual record back, never an error."""
    adapter = flows.get("03")
    built = adapter.build("0.1.0")
    req = json.loads((adapter.project_dir() / "sample_request.json").read_text())
    evidence = evidence_store.capture(adapter, built, req)
    result = overlay_register.stack_off_run(evidence)
    assert result["decision_id"] == evidence.decision_id
    assert isinstance(result["outcome_changed"], bool)
    assert result["counterfactual_record"]["decision_id"] == evidence.decision_id


def test_unwind_estimate_reports_a_rate_over_a_population():
    adapter = flows.get("03")
    built = adapter.build("0.1.0")
    req = json.loads((adapter.project_dir() / "sample_request.json").read_text())
    evidences = [evidence_store.capture(adapter, built, req)]
    result = overlay_register.unwind_estimate("03", evidences)
    assert result["population_size"] == 1
    assert 0.0 <= result["rate"] <= 1.0
