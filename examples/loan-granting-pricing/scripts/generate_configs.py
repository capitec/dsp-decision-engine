"""Write `configs/<version>/`: the synthetic rate card, the four score overlay stacks, the reason codes and the params.

    uv run python examples/loan-granting-pricing/scripts/generate_configs.py [version]
"""
import json
import sys
from pathlib import Path

from credit_core.rate_card import generate_flex_loan_card
from loan_granting.pipeline import build
from loan_granting import reasons

from decider import ConfigurableStep, DecisionTableConfig

BUREAU_STALENESS_DAYS = 45  # Flex Loan's tolerance; the assessment's own default is a week

_FROM = "2026-01-01"


def _overlay(adjustment_id, kind, target, op, value, scope, position, owner, approval, rationale, review_date,
             effective_to=None, tighten_only=True):
    return {"adjustment_id": adjustment_id, "kind": kind, "target": target, "effect": {"op": op, "value": value},
            "scope": scope, "stack_position": position, "owner": owner, "approval_reference": approval,
            "rationale": rationale, "effective_from": _FROM, "effective_to": effective_to, "review_date": review_date,
            "tighten_only": tighten_only}


# Calibration sits between the overlays, so they apply in the order score, scale, probability of default, graded PD.
STACKS = {
    "score_adjustments": ("score", "score_raw", [
        _overlay("ADJ-03-001", "score_shift", "score", "add", -18.0, {"scorecard_id": 1012}, 1, "Model Risk",
                 "MRC-2026-014", "Thin-file defaults running above prediction for two quarters", "2027-01-31",
                 "2027-03-31")]),
    "calibration_scale_adjustments": ("calibration_scale", "scale", [
        _overlay("ADJ-03-002", "scaling_change", "calibration_scale", "multiply", 1.10, {"scorecard_id": 1011}, 2,
                 "Model Risk", "MRC-2026-015",
                 "Points to double the odds widened on the existing-client segment, flattening the tails",
                 "2026-12-31", tighten_only=False)]),
    "probability_of_default_adjustments": ("probability_of_default", "probability_of_default_raw", [
        _overlay("ADJ-03-003", "odds_multiplier", "probability_of_default", "multiply", 1.35, {"channel_code": 4}, 3,
                 "Credit Risk Policy", "CRC-2026-041", "Channel 4 realised defaults running above the model's prediction",
                 "2026-12-31")]),
    # The grade boundary shift is an odds multiplier on the PD that is graded; the risk grade table itself is never edited.
    "probability_of_default_for_grading_adjustments": ("probability_of_default_for_grading", "probability_of_default", [
        _overlay("ADJ-03-004", "odds_multiplier", "probability_of_default_for_grading", "multiply", 1.12,
                 {"product_code": 10}, 4, "Credit Risk Policy", "CRC-2026-042",
                 "The top of the acceptable risk range tightened", "2026-12-31")]),
}

# (code, severity rank with 1 the most severe, description, regulatory)
SCORECARD_REASONS = {4101: "Bureau score", 4102: "Months employed", 4103: "Months in arrears", 4104: "Accounts in arrears",
                     4105: "Revolving utilisation", 4106: "Age", 4107: "Dependants", 4108: "Employment type"}

REASON_ROWS = [
    (reasons.R_FRAUD_DECLINE, 1, "Application-fraud verdict declined", True),
    (reasons.R_EXCLUSION_SANCTIONS, 2, "Sanctions list match", True),
    (reasons.R_EXCLUSION_INTERNAL_FRAUD, 3, "Internal fraud exclusion list match", True),
    (reasons.R_EXCLUSION_STAFF, 4, "Staff restriction list match", True),
    (reasons.R_EXCLUSION_LITIGATION, 5, "Litigation exclusion list match", True),
    (reasons.R_EXCLUSION_OTHER, 6, "Other exclusion list match", True),
    (reasons.R_DECEASED_ESTATE, 7, "Applicant recorded as deceased or under estate administration", True),
    (reasons.R_INSOLVENCY, 8, "Sequestrated and not yet rehabilitated", True),
    (reasons.R_ADMIN_ORDER, 9, "Active administration order", True),
    (reasons.R_DEBT_REVIEW, 10, "Under debt review", True),
    (reasons.R_DUPLICATE_APPLICATION, 11, "Duplicate application within 24 hours", True),
    (reasons.R_IN_FLIGHT_APPLICATION, 12, "An unconcluded Flex Loan or consolidation application exists", True),
    (reasons.R_NO_CAPACITY, 13, "No contractual capacity", True),
    (reasons.R_MAX_AGE_AT_MATURITY, 14, "Would exceed the maximum age at maturity", True),
    (reasons.R_MIN_AGE, 15, "Below the minimum age", True),
    (reasons.R_RESIDENCY, 16, "Residency class not permitted for this product", True),
    (reasons.R_EMPLOYMENT_EXCLUDED, 17, "Employment type not permitted for this product", True),
    (reasons.R_PRODUCT_CHANNEL, 18, "Product not available on this channel", True),
    (reasons.R_BEYOND_APPETITE, 20, "Beyond the Bank's risk appetite", True),
    (reasons.R_CHANNEL_CLOSED, 21, "Channel closed to this product by policy", True),
    (reasons.R_EMPLOYMENT_TENURE_TOO_SHORT, 22, "Employment tenure below the minimum policy threshold", True),
    (reasons.R_ENQUIRY_VELOCITY_EXCESSIVE, 23, "Excessive recent credit enquiries", True),
    (reasons.R_AFFORDABILITY_FAIL, 30, "No amount at any term is affordable", False),
    (reasons.R_AFFORDABILITY_INDETERMINATE, 31, "Affordability could not be determined", True),
    (reasons.R_CONSENT_MISSING, 35, "Bureau-enquiry consent not present or expired", True),
    (reasons.R_EVALUATION_CEILING_REACHED, 36, "The solve reached its evaluation ceiling without a proven maximum", True),
    (reasons.R_BELOW_MIN_AMOUNT, 40, "Offer amount below the product minimum", False),
    (reasons.R_BELOW_MIN_INSTALMENT, 41, "Offer instalment below the minimum viable instalment", False),
    (reasons.R_TOTAL_COST_RATIO_EXCEEDED, 42, "Total cost ratio exceeds the policy threshold", False),
    (reasons.R_IN_DUPLUM_EXCEEDED, 43, "Scheduled charges would exceed the amount advanced (in duplum)", True),
    (reasons.R_EFFECTIVE_RATE_EXCEEDED, 44, "Effective annual rate exceeds the suppression ceiling", False),
    (reasons.R_FINAL_VALIDATION_FAILED, 50, "Final validation could not re-derive the offer", True),
    *[(code, 60 + code - 4100, f"Low score: {name}", False) for code, name in SCORECARD_REASONS.items()],
]

REASON_CODES = {
    "type": "reason_codes", "name": "rank_reasons", "version": "loan-granting-reasons-2026.09",
    "codes": [{"code": code, "severity_rank": rank, "description": description, "is_regulatory": regulatory}
              for code, rank, description, regulatory in REASON_ROWS],
}


def stack_document(name: str) -> dict:
    target, base_field, overlays = STACKS[name]
    return {"type": "adjustment_stack", "name": f"apply_{target}_adjustments", "target": target,
            "set_id": "AS-03-2026.09", "base_field": base_field, "adjustments": overlays}


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    card = generate_flex_loan_card("rc-2026.09")
    (out / "rate_card_flex_loan.json").write_text(json.dumps(card, separators=(",", ":")))
    documents = {name: stack_document(name) for name in STACKS}
    for name, document in documents.items():
        (out / f"{name}.json").write_text(json.dumps(document, indent=2) + "\n")
    (out / "reason_codes.json").write_text(json.dumps(REASON_CODES, indent=2) + "\n")
    stacks = {name: ConfigurableStep.load(document) for name, document in documents.items()}
    params = build(DecisionTableConfig.load(card), **stacks, reason_codes=ConfigurableStep.load(REASON_CODES)
                   ).parameters().defaults()
    params["loan_granting"]["affordability_assessment"]["evidence"]["bureau_is_stale"]["staleness_window_days"] = (
        BUREAU_STALENESS_DAYS)
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {len(card['rows'])} rate card rows, {len(documents)} overlay stacks, the reason codes and {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
