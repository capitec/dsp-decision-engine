"""Write `configs/<version>/`: the product rate cards, the settleability rules and the params document.

    uv run python examples/consolidation/scripts/generate_configs.py [version]

Every table and rate is synthetic. Rates in the cards are percentages, as Treasury would author them.
"""
import json
import sys
from pathlib import Path

from credit_core.evidence import cell_id

from consolidation.pipeline import build
from decider import DecisionTableConfig

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


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    documents = {"rate_card_product11": product11_card(), "card_promotional_rates": card_promotional_card(),
                 "card_reversion_rates": card_reversion_card(), "settleability_rules": settleability_rules()}
    for name, document in documents.items():
        (out / f"{name}.json").write_text(json.dumps(document, separators=(",", ":")))
    params = build(*(DecisionTableConfig.load(d) for d in documents.values())).parameters().defaults()
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {len(documents)} documents and {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
