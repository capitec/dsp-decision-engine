"""Write `configs/<version>/`: the four overlay documents, the reason codes, and `params.json` with every policy number at its default,
this product's fee and premium terms and the Flex Loan rate card.

    uv run python examples/retail-credit-e2e/scripts/generate_configs.py [version]
"""
import json
import re
import sys
from pathlib import Path

from credit_core import eligibility as core_eligibility
from decider import ConfigurableStep

from retail_credit import caps, disclosure, eligibility, routing, validation
from retail_credit.flex_card import card_tables
from retail_credit.pipeline import build

# The shared fee and premium steps default to other products' terms. Overrides are by step name, because two steps can
# share a parameter name and matching on the name alone would patch the wrong one.
def overlay(adjustment_id, kind, target, op, value, scope, approval, rationale, effective_to):
    return {
        "adjustment_id": adjustment_id, "kind": kind, "target": target, "effect": {"op": op, "value": value},
        "scope": scope, "stack_position": 1, "owner": "Credit Risk Policy", "approval_reference": approval,
        "rationale": rationale, "effective_from": "2026-01-01", "effective_to": effective_to,
        "review_date": "2027-06-30", "tighten_only": True}


def stack(target, base_field, adjusted_output, unadjusted_output, adjustments):
    return {"type": "adjustment_stack", "name": f"apply_{target}_adjustments", "target": target, "set_id": "AS-2026.09",
            "base_field": base_field, "adjusted_output": adjusted_output, "unadjusted_output": unadjusted_output,
            "adjustments": adjustments}


# One document per overlaid value: the file name is the argument of `build` that receives it.
OVERLAYS = {
    "pd_overlays": stack("probability_of_default", "probability_of_default_raw", None, "probability_of_default_unadjusted", [
        overlay("ADJ-0061", "odds_multiplier", "probability_of_default", "multiply", 1.18, {"channel_code": 3}, "CC-2027-04",
                "Channel 3 defaults above model prediction for two quarters", "2027-09-30")]),
    "amount_cap_overlays": stack("amount_cap", "amount_cap_before_overlay", "amount_cap", "amount_cap_unadjusted", [
        overlay("ADJ-0087", "cap_adjustment", "amount_cap", "multiply", 0.90, {"channel_code": 3}, "CC-2027-11",
                "Channel 3 amount cap tightened pending a portfolio review", "2027-12-31")]),
    "buffer_overlays": stack("affordability_buffer", "buffer", "affordability_buffer_applied", "affordability_buffer_unadjusted", [
        overlay("ADJ-0114", "buffer_adjustment", "affordability_buffer", "add", 0.04, {"channel_code": 3}, "CC-2027-07",
                "Channel 3 buffer raised pending an affordability review", "2027-12-31")]),
    "rate_addon_overlays": stack("nominal_annual_rate", "rate_addon_base", "rate_addon_bps", "rate_addon_unadjusted", [
        overlay("ADJ-0132", "rate_addon", "nominal_annual_rate", "add", 0.006, {"segment_code": 4}, "CC-2027-13",
                "60 basis point add-on for segment 4 pending the next card refresh", "2027-12-31")]),
}

# code, severity rank (1 is the most severe), description, whether the reason is regulatory
REASON_CODES = {
    "type": "reason_codes", "name": "rank_reasons", "version": "reasons-2026.09",
    "codes": [
        {"code": code, "severity_rank": rank, "description": description, "is_regulatory": regulatory}
        for code, rank, description, regulatory in [
            (core_eligibility.R_UNDERAGE, 10, "Applicant below minimum age", True),
            (core_eligibility.R_NOT_RESIDENT, 15, "Applicant not resident", True),
            (core_eligibility.R_NO_CAPACITY, 5, "No contractual capacity", True),
            (core_eligibility.R_PRODUCT_UNAVAILABLE, 40, "Product not available to applicant", False),
            (core_eligibility.R_EXCLUSION_LIST, 8, "On an internal exclusion list", True),
            (core_eligibility.R_SANCTIONED, 1, "Sanctioned party", True),
            (core_eligibility.R_DECEASED, 2, "Deceased on record", True),
            (core_eligibility.R_DEBT_REVIEW, 12, "Under debt review", True),
            (core_eligibility.R_ADMINISTRATION, 12, "Under administration", True),
            (eligibility.R_AGE_AT_MATURITY, 11, "Age at maturity exceeds the product maximum", True),
            (eligibility.R_IN_FLIGHT_DUPLICATE, 30, "A duplicate application is already in flight", False),
            (eligibility.R_BUREAU_CONSENT_MISSING, 3, "Bureau enquiry consent not given", True),
            (routing.R_AMOUNT_OUT_OF_RANGE, 41, "Requested amount outside the product's range", False),
            (routing.R_TERM_OUT_OF_RANGE, 42, "Requested term outside the product's range", False),
            (routing.R_GRADE_WORSE_THAN_ACCEPTABLE, 20, "Risk grade worse than the policy caps allow", False),
            (caps.R_CHANNEL_CLOSED, 44, "Channel not open for this product", False),
            (caps.R_APPETITE_DECLINE, 21, "Risk grade outside the Bank's appetite", False),
            (caps.R_ENQUIRY_VELOCITY_DECLINE, 22, "Bureau enquiry velocity too high", False),
            (disclosure.R_AFFORDABILITY_FAIL, 20, "Proposed instalment exceeds affordable capacity", True),
            (disclosure.R_NO_AFFORDABLE_AMOUNT, 26,
             "No amount at or above the product minimum fits the instalment ceiling", False),
            (disclosure.R_FRAUD_DECLINE, 1, "Application fraud verdict: decline", True),
            (disclosure.R_FRAUD_REFER, 6, "Application fraud verdict: refer", True),
            (disclosure.R_VALIDATION_FAILED, 27, "Offer failed final validation", False),
            (validation.A_AMOUNT_ON_GRID, 50, "Offer amount not on the R250 grid", False),
            (validation.A_AMOUNT_WITHIN_CAP, 23, "Offer amount exceeds the cap in force", False),
            (validation.A_TERM_WITHIN_CAP, 24, "Offer term exceeds the cap in force", False),
            (validation.A_RATE_WITHIN_CEILING, 4, "Rate exceeds the statutory ceiling", True),
            (validation.A_INSTALMENT_AFFORDABLE, 20, "Offer instalment exceeds affordable capacity", True),
        ]
    ],
}

STEP_OVERRIDES = {
    "initiation_fee": {"base_fee": 210.0, "marginal_rate": 0.095, "threshold": 1200.0},
    "credit_life_premium": {"cap": 1000.0},
}


def fill(node: dict, tables: dict) -> None:
    """Patch every step of `node` named in `STEP_OVERRIDES`, and put the card's tables where `card_rate` reads them."""
    for name, value in node.items():
        if name in STEP_OVERRIDES:
            value.update(STEP_OVERRIDES[name])
        elif name == "card_rate":
            value.update(tables)
        elif isinstance(value, dict):
            fill(value, tables)


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    documents = {**OVERLAYS, "reason_codes": REASON_CODES}
    for name, document in documents.items():
        (out / f"{name}.json").write_text(json.dumps(document, indent=2) + "\n")
    params = build(**{name: ConfigurableStep.load(document) for name, document in documents.items()}).parameters().defaults()
    fill(params, card_tables())
    text = json.dumps(params, indent=2)
    text = re.sub(r'\{\s+"(rate|floor)": ([^\s,}]+)\s+\}', r'{"\1": \2}', text)  # one card row per line
    (out / "params.json").write_text(text + "\n")
    print(f"wrote {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
