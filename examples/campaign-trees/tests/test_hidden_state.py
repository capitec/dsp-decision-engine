"""Everything the decision reads is in a document, a param or a step's fields, so a release check can compare it."""
import json
import random
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
from decider import ConfigurableStep, Engine
from decider.exceptions import HiddenStateWarning
from decider.lab import diff

from loan_granting.pricing import CreditLifeIndex, RateCardIndex

from campaign_trees import preassessment
from campaign_trees.pipeline import build

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "0.1.0"


def cap_document(**changes):
    document = json.loads((CONFIGS / "cap_reductions.json").read_text())
    document["adjustments"][0].update(changes)
    return ConfigurableStep.load(document)


def test_building_the_pipeline_reads_no_hidden_state(tree, rate_card, cap_reductions):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Engine().bind(build(tree, rate_card, cap_reductions), mode="interpreted")
    assert [str(w.message) for w in caught if issubclass(w.category, HiddenStateWarning)] == []


def test_a_changed_cap_is_a_different_pipeline_and_a_new_rationale_is_not(tree, rate_card, cap_reductions):
    base = build(tree, rate_card, cap_reductions)
    assert diff.structure(base, build(tree, rate_card, cap_document())).identical
    assert diff.structure(base, build(tree, rate_card, cap_document(rationale="Reworded"))).identical
    assert not diff.structure(base, build(tree, rate_card, cap_document(effect={"op": "multiply", "value": 0.6}))).identical


def as_table(rows, columns):
    return namedtuple("Table", columns)(*(np.array([row[c] for row in rows]) for c in columns))


def test_the_table_lookups_equal_the_granting_example_indexes(rate_card):
    card = preassessment._Card(as_table(preassessment.card_rows(rate_card), preassessment.CARD_COLUMNS))
    life = preassessment._CreditLife(as_table(preassessment.credit_life_rows(), preassessment.CREDIT_LIFE_COLUMNS))
    index, life_index = RateCardIndex.from_card(rate_card), CreditLifeIndex()
    rng = random.Random(3)
    for _ in range(300):
        amount, term, grade = rng.uniform(0, 600_000), float(rng.randint(1, 100)), rng.randint(0, 13)
        assert (card.lookup(amount, term, grade) or [None])[0] == (index.lookup(amount, term, grade) or [None])[0]
        assert card.amount_bands(term, grade) == index.amount_bands(term, grade)
        age, joint = rng.uniform(15, 90), rng.random() < 0.5
        assert life.lookup(age, term, joint)[0] == life_index.lookup(age, term, joint)[0]
