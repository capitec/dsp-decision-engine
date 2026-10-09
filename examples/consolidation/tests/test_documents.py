"""The reason codes and overlays are config documents: editing one changes the decision, with no code change."""
import json
from pathlib import Path

from assessment.documents import shipped as assessment_document
from decider import ConfigurableStep, DecisionTableConfig

from consolidation.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
ASSESSMENT = {"capacity_adjustments": "capacity_adjustments", "affordability_reason_codes": "reason_codes"}  # shipped by `assessment`
TABLES = ("rate_card_product11", "card_promotional_rates", "card_reversion_rates", "settleability_rules")


def raw(name):
    return json.loads((ROOT / "configs" / "0.1.0" / f"{name}.json").read_text())


def rebuilt(**edits):
    """The pipeline built from the shipped documents, with `edit(document)` applied to each document named."""
    documents = {name: DecisionTableConfig.load(raw(name)) for name in TABLES}
    for name, edit in edits.items():
        document = json.loads(assessment_document(ASSESSMENT[name]).model_dump_json()) if name in ASSESSMENT else raw(name)
        edit(document)
        documents[name] = (DecisionTableConfig if name in TABLES else ConfigurableStep).load(document)
    return build(**documents).compile("interpreted")


def severity(code, rank):
    def edit(document):
        next(c for c in document["codes"] if c["code"] == code)["severity_rank"] = rank

    return edit


def test_a_client_reason_severity_edit_changes_which_decline_reason_is_primary(sample, params, parse):
    declined = parse(sample | {"client_under_administration": True, "bureau_unobtainable": True})
    assert rebuilt().score(declined, params)["primary_reason_code"] == 2002
    out = rebuilt(client_reason_codes=severity(2004, 1)).score(declined, params)
    assert (out["decline_reason_codes"], out["primary_reason_code"]) == ([2004, 2002], 2004)


def test_a_scenario_rejection_severity_edit_changes_which_rejection_is_primary(sample, params, parse):
    def primaries(pipeline):
        out = pipeline.score(parse(sample), params)
        return {s["scenario_id"]: s["primary_rejection_reason_code"] for s in out["scenarios"] if {1003, 1004} <= set(s["rejection_reason_codes"])}

    shipped, edited = primaries(rebuilt()), primaries(rebuilt(scenario_rejection_codes=severity(1003, 1)))
    assert shipped and set(shipped.values()) == {1004} and set(edited.values()) == {1003}


def test_an_intervention_overlay_edit_moves_the_anti_harm_threshold_of_product_11(sample, params, parse):
    def ceiling(pipeline):
        scenarios = pipeline.score(parse(sample), params)["scenarios"]
        flex = next(s for s in scenarios if s["product_code"] == 11 and s["interventions"])
        return next(i for i in flex["interventions"] if i["intervention"] == 4)

    halved = ceiling(rebuilt(intervention_adjustments=lambda doc: doc["adjustments"][0]["effect"].update(value=0.5)))
    assert (halved["overlay_id"], halved["base_threshold"], halved["threshold"]) == ("ADJ-CON-INT-2026-001", 0.15, 0.075)


def test_an_objective_overlay_edit_changes_the_weights_the_branch_channel_is_scored_with(sample, params, parse):
    def reweigh(doc):
        doc["adjustments"][0]["effect"]["value"] = 0.2

    out = rebuilt(objective_adjustments=reweigh).score(parse(sample | {"channel_code": 1}), params)
    assert out["objective_weights"]["min_commitment"] == 0.333333 and out["objective_weights"]["new_money"] == 0.666667


def test_a_rate_card_document_edit_reprices_the_consolidation_loan(sample, params, parse):
    def dearer(document):
        for row in document["rows"]:
            row["rate"] += 1.0

    base = rebuilt().score(parse(sample), params)["scenarios"]
    edited = rebuilt(rate_card_product11=dearer).score(parse(sample), params)["scenarios"]
    flex = [(a, b) for a, b in zip(base, edited) if a["product_code"] == 11 and "nominal_annual_rate" in a]
    assert flex and all(round(b["nominal_annual_rate"] - a["nominal_annual_rate"], 4) == 0.01 for a, b in flex)
    assert all(a["nominal_annual_rate"] == b["nominal_annual_rate"] for a, b in zip(base, edited) if a["product_code"] == 20)


def test_a_capacity_overlay_edit_moves_the_capacity_every_scenario_is_tested_against(sample, params, parse):
    def tighten(document):
        document["adjustments"][0].update(scope={"product_code": 11})
        document["adjustments"][0]["effect"]["value"] = 0.5

    record = parse(sample)
    shipped = rebuilt().score(record, params)
    tightened = rebuilt(capacity_adjustments=tighten).score(record, params)
    assert tightened["baseline"]["max_affordable_instalment"] < shipped["baseline"]["max_affordable_instalment"]
    assert tightened["scenarios"][0]["max_affordable_instalment"] < shipped["scenarios"][0]["max_affordable_instalment"]
