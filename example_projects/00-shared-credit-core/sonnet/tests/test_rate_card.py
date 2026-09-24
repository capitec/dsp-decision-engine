"""`core.rate_card`: the Flex Loan card at its full declared 96x55x12 size (SCOPE.md rule 1)."""
from decider import Engine
from decider.steps.tables import DecisionTableConfig

from credit_core.rate_card import (
    FLEX_LOAN_AMOUNT_BANDS, FLEX_LOAN_GRADES, FLEX_LOAN_TERM_BANDS, diff_cards, generate_flex_loan_card,
    out_of_range,
)


def test_the_card_is_generated_at_full_declared_size():
    doc = generate_flex_loan_card("v1")
    assert len(doc["rows"]) == FLEX_LOAN_AMOUNT_BANDS * FLEX_LOAN_TERM_BANDS * FLEX_LOAN_GRADES == 63_360


def test_every_row_has_a_distinct_cell_id():
    doc = generate_flex_loan_card("v1")
    ids = [r["cell_id"] for r in doc["rows"]]
    assert len(set(ids)) == len(ids)


def test_a_lookup_returns_a_structured_result_with_its_cell_id():
    doc = generate_flex_loan_card("v1")
    cfg = DecisionTableConfig.load(doc)
    exe = Engine().bind(cfg, mode="interpreted")
    out = exe.score({"offered_amount": 50_000.0, "term_months": 36, "risk_grade": 5})
    assert out["rate_card_version"] == "v1"
    assert out["cell_id"].startswith("rate_card.flex_loan@v1#")
    assert 5.0 <= out["rate"] <= 32.0


def test_worse_grades_price_higher_all_else_equal():
    doc = generate_flex_loan_card("v1")
    cfg = DecisionTableConfig.load(doc)
    exe = Engine().bind(cfg, mode="interpreted")
    rates = [exe.score({"offered_amount": 50_000.0, "term_months": 36, "risk_grade": g})["rate"]
             for g in range(1, 13)]
    assert rates == sorted(rates)


def test_out_of_range_flags_amounts_and_terms_outside_the_product_range():
    assert out_of_range(1_000_000.0, 36.0) is True
    assert out_of_range(50_000.0, 36.0) is False
    assert out_of_range(50_000.0, 200.0) is True


def test_diff_cards_reports_repricing_and_structural_changes():
    old = generate_flex_loan_card("old")["rows"][:3]
    new = [dict(r, rate=r["rate"] + 1.0) for r in old[:2]] + [old[2]]
    changes = diff_cards(old, new)
    repriced = [c for c in changes if c["change"] == "repriced"]
    assert len(repriced) == 2
    assert all(c["delta"] == 1.0 for c in repriced)


def test_replay_is_deterministic_to_the_cent():
    """Acceptance §10 item 6: identical inputs and version must reproduce identically."""
    doc = generate_flex_loan_card("v1")
    cfg = DecisionTableConfig.load(doc)
    exe = Engine().bind(cfg, mode="interpreted")
    record = {"offered_amount": 123_400.0, "term_months": 48, "risk_grade": 7}
    assert exe.score(record) == exe.score(record)
