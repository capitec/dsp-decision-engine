"""Write `configs/<version>/`: the product rate cards, the settleability rules, the reason codes, the overlays and the params document.

    uv run python examples/consolidation/scripts/generate_configs.py [version]

Every table and rate is synthetic. Rates in the cards are percentages, as Treasury would author them.
"""
import json
import sys
from pathlib import Path

from credit_core.evidence import cell_id
from decikit import ConfigurableStep, DecisionTableConfig

from consolidation.pipeline import build

INF = float("inf")


def bands(lo: float, hi: float, n: int) -> list[tuple[float, float]]:
    """`n` contiguous bands across `[lo, hi]` with infinite outer edges, so an amount outside the product's range
    still lands in a band (the product's own limits reject it)."""
    width = (hi - lo) / n
    return [(-INF if i == 0 else lo + i * width, INF if i == n - 1 else lo + (i + 1) * width) for i in range(n)]


def between(variable: str, lo: str, hi: str) -> dict:
    return {"type": "between", "variable": variable, "lower_bound_column": lo, "upper_bound_column": hi, "allow_gaps": True}


def product11_card(version: str = "rc-p11-2026.09") -> dict:
    """Flex Loan Consolidation: 12 amount bands x 12 term bands x 12 grades, one absolute rate per cell."""
    rows = [
        {"amt_lo": a_lo, "amt_hi": a_hi, "term_lo": t_lo, "term_hi": t_hi, "grade": grade,
         "rate": round(min(max(9.0 + (grade - 1) * 1.3 + ai * 0.05 - ti * 0.03, 8.5), 29.5), 4),
         "cell_id": cell_id("rate_card.product11", version, ai, ti, grade), "rate_card_version": version}
        for ai, (a_lo, a_hi) in enumerate(bands(10_000.0, 500_000.0, 12))
        for ti, (t_lo, t_hi) in enumerate(bands(12.0, 84.0, 12))
        for grade in range(1, 13)]
    return {"type": "decision_table", "name": "rate_card_product11",
            "columns": {"amt_lo": "Float64", "amt_hi": "Float64", "term_lo": "Float64", "term_hi": "Float64",
                        "grade": "Int64", "rate": "Float64", "cell_id": "String", "rate_card_version": "String"},
            "rows": rows,
            "expression": {"type": "and", "expressions": [
                between("offered_amount", "amt_lo", "amt_hi"), between("term_months", "term_lo", "term_hi"),
                {"type": "eq", "variable": "risk_grade", "value_column": "grade"}]},
            "outputs": ["rate", "cell_id", "rate_card_version"], "default": [None, None, None]}


def card_promotional_card(version: str = "rc-p20-promo-2026.09") -> dict:
    """Everyday Card balance transfer, promotional rate: 8 limit bands x 5 durations x 12 grades."""
    rows = [
        {"lim_lo": lo, "lim_hi": hi, "duration": duration, "grade": grade,
         "rate": round(min(max(6.0 + (grade - 1) * 0.9 + li * 0.1 - duration * 0.05, 0.0), 22.0), 4),
         "cell_id": cell_id("rate_card.card_promotional", version, li, duration, grade)}
        for li, (lo, hi) in enumerate(bands(1_000.0, 300_000.0, 8))
        for duration in (0, 6, 12, 18, 24) for grade in range(1, 13)]
    return {"type": "decision_table", "name": "card_promotional_rates",
            "columns": {"lim_lo": "Float64", "lim_hi": "Float64", "duration": "Int64", "grade": "Int64",
                        "rate": "Float64", "cell_id": "String"},
            "rows": rows,
            "expression": {"type": "and", "expressions": [
                between("approved_limit", "lim_lo", "lim_hi"),
                {"type": "eq", "variable": "promotional_months", "value_column": "duration"},
                {"type": "eq", "variable": "risk_grade", "value_column": "grade"}]},
            "outputs": ["rate", "cell_id"], "default": [None, None]}


def card_reversion_card(version: str = "rc-p20-reversion-2026.09") -> dict:
    """Everyday Card reversion rate, charged once the promotional period ends: 16 limit bands x 12 grades."""
    rows = [
        {"lim_lo": lo, "lim_hi": hi, "grade": grade,
         "rate": round(min(max(21.0 + (grade - 1) * 1.1 - li * 0.06, 15.0), 32.0), 4),
         "cell_id": cell_id("rate_card.card_reversion", version, li, grade)}
        for li, (lo, hi) in enumerate(bands(1_000.0, 300_000.0, 16)) for grade in range(1, 13)]
    return {"type": "decision_table", "name": "card_reversion_rates",
            "columns": {"lim_lo": "Float64", "lim_hi": "Float64", "grade": "Int64", "rate": "Float64", "cell_id": "String"},
            "rows": rows,
            "expression": {"type": "and", "expressions": [
                between("approved_limit", "lim_lo", "lim_hi"),
                {"type": "eq", "variable": "risk_grade", "value_column": "grade"}]},
            "outputs": ["rate", "cell_id"], "default": [None, None]}


def settleability_rules(version: str = "settleability-2026.09") -> dict:
    """What each known provider does with a third-party settlement request. A provider not listed is unknown."""
    providers = [  # code, quotes to third parties, quotation turnaround days, security release days, early settlement rule
        (0, 1, 0, 0, 0),  # the Bank itself: instant
        (101, 1, 2, 10, 1), (102, 1, 2, 10, 1),  # on the electronic exchange
        (103, 1, 5, 10, 1), (105, 1, 5, 10, 1), (106, 1, 5, 10, 1),
        (201, 1, 10, 10, 2),  # still posts its quotations
        (901, 0, 0, 0, 0), (902, 0, 0, 0, 0), (903, 0, 0, 0, 0),  # do not quote third parties at all
    ]
    rows = [{"provider_code": p, "quotes_to_third_parties": q, "quotation_turnaround_days": t, "security_release_days": r,
             "early_settlement_rule_code": e, "cell_id": cell_id("settleability", version, p)}
            for p, q, t, r, e in providers]
    return {"type": "decision_table", "name": "settleability_rules",
            "columns": {"provider_code": "Int64", "quotes_to_third_parties": "Int64", "quotation_turnaround_days": "Int64",
                        "security_release_days": "Int64", "early_settlement_rule_code": "Int64", "cell_id": "String"},
            "rows": rows,
            "expression": {"type": "eq", "variable": "provider_code", "value_column": "provider_code"},
            "outputs": ["quotes_to_third_parties", "quotation_turnaround_days", "security_release_days",
                        "early_settlement_rule_code", "cell_id"],
            "default": [0, 0, 0, 0, None]}


# (code, severity rank with 1 the most severe, description, regulatory). A policy intervention CON-INT-n is 1000 + n.
SCENARIO_REJECTIONS = [
    (1001, 21, "More accounts settled than the product permits", True),
    (1003, 23, "Instalment reduction below the product's minimum", True),
    (1004, 14, "Total cost of credit rises more than the anti-harm ceiling allows", True),
    (1005, 25, "Term extends too far beyond the settled accounts' longest remaining term", True),
    (1006, 26, "New rate is above the settled accounts' weighted average rate", True),
    (1008, 28, "New money above the permitted share of the settlement total or the absolute cap", True),
    (1009, 29, "Post-consolidation debt service ratio above the ceiling", True),
    (1010, 30, "Too little of the advance goes to external creditors", True),
    (1013, 33, "The client has reached the maximum consolidations in 24 months", True),
    (1014, 34, "Discretionary income after the new instalment is below the floor", True),
    (1101, 90, "The product's rate card has no cell for this amount, term and grade", False),
    (1102, 5, "The new instalment exceeds what the client can afford", True),
    (1103, 80, "The amount or limit is outside the product's limits", False),
]
CLIENT_REASONS = [
    (2001, 1, "The client is under debt review", True),
    (2002, 2, "The client is under administration or sequestration", True),
    (2003, 5, "No verified income at the decision date", True),
    (2004, 6, "No bureau view could be obtained", True),
    (2006, 15, "An active reckless lending allegation is on file", True),
    (2005, 20, "A consolidation concluded in the last 6 months", True),
    (2007, 25, "An account the client asked to settle cannot be settled", False),
    (2008, 40, "No scenario passed every intervention", True),
    (2010, 7, "Affordability could not be established on the evidence", True),
    (2009, 45, "The request is not affordable as made and consolidation does not apply", True),
]


def overlay(adjustment_id: str, kind: str, target: str, op: str, value: float, scope: dict, owner: str, approval: str,
            rationale: str, **fields) -> dict:
    return {"adjustment_id": adjustment_id, "kind": kind, "target": target, "effect": {"op": op, "value": value},
            "scope": scope, "stack_position": 1, "owner": owner, "approval_reference": approval, "rationale": rationale,
            "effective_from": "2026-01-01", "effective_to": None, "review_date": "2026-12-31", **fields}


# Overlays read inside the search and the selection, which is why they are `overlay_set` documents, not stacks in a flow.
OVERLAYS = {
    "intervention_adjustments": {"type": "overlay_set", "name": "intervention_adjustments", "set_id": "AS-CON-INT-2026.09",
                                 "adjustments": [overlay(
        "ADJ-CON-INT-2026-001", "cap_adjustment", "anti_harm", "multiply", 0.80, {"product_code": 11}, "Credit Risk Policy",
        "CRC-2026-044", "Anti-harm ceiling tightened on product 11 pending the ombud review", tighten_only=True)]},
    "objective_adjustments": {"type": "overlay_set", "name": "objective_adjustments", "set_id": "AS-CON-OBJ-2026.09",
                              "adjustments": [overlay(
        f"ADJ-CON-OBJ-2026-00{n}", "objective_reweight", f"weight_{objective}", "set", value, {"channel_code": 1},
        "Credit Committee", "CC-2026-014", "Branch channel weighted towards instalment relief this quarter")
        for n, (objective, value) in enumerate((("min_commitment", 0.6), ("new_money", 0.4)), start=1)]},
}


def reason_codes(name: str, version: str, rows: list[tuple]) -> dict:
    return {"type": "reason_codes", "name": name, "version": version,
            "codes": [{"code": c, "severity_rank": r, "description": d, "is_regulatory": reg} for c, r, d, reg in rows]}


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    documents = {"rate_card_product11": product11_card(), "card_promotional_rates": card_promotional_card(),
                 "card_reversion_rates": card_reversion_card(), "settleability_rules": settleability_rules()}
    for name, document in documents.items():
        (out / f"{name}.json").write_text(json.dumps(document, separators=(",", ":")))
    codes = {"client_reason_codes": reason_codes("client_reason_codes", "consolidation-client-2026.09", CLIENT_REASONS),
             "scenario_rejection_codes": reason_codes("scenario_rejection_codes", "consolidation-rejections-2026.09",
                                                      SCENARIO_REJECTIONS)}
    codes |= OVERLAYS
    for name, document in codes.items():
        (out / f"{name}.json").write_text(json.dumps(document, indent=2) + "\n")
    params = build(*(DecisionTableConfig.load(d) for d in documents.values()),
                   **{name: ConfigurableStep.load(d) for name, d in codes.items()}).parameters().defaults()
    # The rate card documents are the one place a card is edited; the table params only default to their rows.
    for key in ("rate_card_product11", "rate_card_product11_version"):
        del params["shared"][key]
    for key in ("plain_rate_card", "plain_rate_card_version"):
        del params["consolidation"]["plain_granting"][key]
    for key in ("card_promotional_rates", "card_promotional_rates_version", "card_reversion_rates", "card_reversion_rates_version"):
        del params["consolidation"]["search"][key]
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {len(documents) + len(codes)} documents and {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
