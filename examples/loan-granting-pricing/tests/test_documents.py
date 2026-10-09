"""What a release check sees of the pipeline: the overlays, reason codes and rate card are documents, not objects."""
import copy
import json
import random
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
from decider import ConfigurableStep, DecisionTableConfig
from decider.lab import diff

from loan_granting import reasons
from loan_granting.pipeline import build
from loan_granting.pricing import CreditLifeIndex, RateCardIndex
from loan_granting.rate_tables import CreditLifeTable, RateCardTable, credit_life_rows, rate_card_rows

ROOT = Path(__file__).resolve().parents[1]


def columns(rows):
    return namedtuple("Table", rows[0])(*(np.array([r[c] for r in rows]) for c in rows[0]))


def document(name, edit=lambda doc: None):
    doc = json.loads((ROOT / "configs" / "0.1.0" / f"{name}.json").read_text())
    edit(doc)
    return ConfigurableStep.load(doc)


def rebuilt(documents, **edits):
    """The pipeline built from the shipped documents, with `edit(doc)` applied to the named ones."""
    return build(**{**documents, **{name: document(name, edit) for name, edit in edits.items()}})


def changed(old, new):
    """The steps that differ, apart from the embedded assessment's own (the assessment is compared in its own example)."""
    return [c for c in diff.structure(old, new).changed_steps if "/affordability_assessment/" not in c.ref.path]


@pytest.fixture(scope="module")
def shipped(documents):
    return build(**documents)


def test_the_same_documents_build_an_unchanged_pipeline(shipped, documents):
    assert changed(shipped, build(**documents)) == []


def test_doubling_a_score_overlay_changes_the_overlay_step(shipped, documents):
    doubled = rebuilt(documents, score_adjustments=lambda doc: doc["adjustments"][0]["effect"].update(value=-36.0))
    steps = changed(shipped, doubled)
    assert [c.ref.path for c in steps] == ["loan_granting/scoring/apply_score_adjustments"] and steps[0].consts


def test_a_governance_only_edit_to_an_overlay_is_not_a_change(shipped, documents):
    def reword(doc):
        doc["adjustments"][0].update(rationale="reworded", owner="Credit Risk Policy", review_date="2027-03-31")

    assert changed(shipped, rebuilt(documents, probability_of_default_adjustments=reword)) == []


def test_changing_a_reason_severity_changes_the_ranking_step(shipped, documents):
    reranked = rebuilt(documents, reason_codes=lambda doc: doc["codes"][0].update(severity_rank=99))
    assert [c.ref.path for c in changed(shipped, reranked)] == ["loan_granting/decision/rank_reasons"]


def test_a_different_rate_card_document_changes_the_steps_that_price(shipped, documents):
    card = json.loads((ROOT / "configs" / "0.1.0" / "rate_card_flex_loan.json").read_text())
    card["rows"][0]["rate"] += 1.0
    dearer = build(**{**documents, "rate_card_flex_loan": DecisionTableConfig.load(card)})
    assert {c.ref.path for c in changed(shipped, dearer)} == {"loan_granting/solve_terms", "loan_granting/final_validation"}


def test_every_reason_code_the_product_gives_is_in_the_reason_codes_document():
    in_document = {c["code"] for c in json.loads((ROOT / "configs" / "0.1.0" / "reason_codes.json").read_text())["codes"]}
    given = {value for name, value in vars(reasons).items() if name.startswith("R_")}
    assert given <= in_document


def test_a_different_rate_card_in_the_params_document_prices_the_offer(pipeline, sample, params, documents, parse):
    rows, _ = rate_card_rows(documents["rate_card_flex_loan"])
    dearer = copy.deepcopy(params)
    dearer["shared"]["rate_card"] = [{**row, "rate": row["rate"] + 1.0} for row in rows]
    dearer["shared"]["rate_card_version"] = "rc-test"
    before = pipeline.score(parse(sample), params)["recommended_offer"]
    after = pipeline.score(parse(sample), dearer)["recommended_offer"]
    assert after["rate_card_version"] == "rc-test"
    assert after["nominal_annual_rate"] == pytest.approx(before["nominal_annual_rate"] + 0.01)


def test_a_rate_card_whose_cell_ids_are_not_its_bands_is_refused():
    document = json.loads((ROOT / "configs" / "0.1.0" / "rate_card_flex_loan.json").read_text())
    document["rows"][0]["cell_id"] = "somewhere-else"
    with pytest.raises(ValueError, match="cell ids must be"):
        rate_card_rows(DecisionTableConfig.load(document))


def test_the_table_lookup_answers_as_the_rate_card_index_does(rate_card):
    rows, version = rate_card_rows(rate_card)
    table, index = RateCardTable(columns(rows), version), RateCardIndex.from_card(rate_card)
    rng = random.Random(3)
    for _ in range(300):
        amount, term, grade = rng.uniform(500, 600_000), float(rng.randint(3, 90)), rng.randint(0, 14)
        assert table.lookup(amount, term, grade) == index.lookup(amount, term, grade)
    for term, grade in [(6, 1), (60, 9), (84.0, 12), (3, 5), (36, 99)]:
        old, new = index.amount_bands(term, grade), table.amount_bands(term, grade)
        assert [r for _, r in new] == [r for _, r in old] and [s for s, _ in new][1:] == [s for s, _ in old][1:]


def test_the_credit_life_table_lookup_answers_as_the_credit_life_index_does():
    rows = credit_life_rows()
    lookup, index = CreditLifeTable(columns(rows), "cl-2026.09"), CreditLifeIndex()
    for age in (18.0, 29.9, 30.0, 54.0, 55.0, 80.0):
        for term in (6, 24, 59, 60, 84):
            for joint in (False, True):
                assert lookup.lookup(age, term, joint) == index.lookup(age, term, joint)
