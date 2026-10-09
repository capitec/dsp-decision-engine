"""Write `configs/<version>/`: the four overlay documents, and `params.json` with every policy number at its default,
this product's fee and premium terms and the Flex Loan rate card.

    uv run python examples/retail-credit-e2e/scripts/generate_configs.py [version]
"""
import json
import re
import sys
from pathlib import Path

from decider import ConfigurableStep

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
    for name, document in OVERLAYS.items():
        (out / f"{name}.json").write_text(json.dumps(document, indent=2) + "\n")
    params = build(**{name: ConfigurableStep.load(document) for name, document in OVERLAYS.items()}).parameters().defaults()
    fill(params, card_tables())
    text = json.dumps(params, indent=2)
    text = re.sub(r'\{\s+"(rate|floor)": ([^\s,}]+)\s+\}', r'{"\1": \2}', text)  # one card row per line
    (out / "params.json").write_text(text + "\n")
    print(f"wrote {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
