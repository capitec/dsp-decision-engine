"""Product 11's card and the credit life rates are table params that price exactly as the index objects they replace."""
import itertools
import json
from pathlib import Path

import pytest
from decider import DecisionTableConfig, param_table
from loan_granting.pricing import CreditLifeIndex, RateCardIndex
from loan_granting.rate_tables import RATE_CARD_COLUMNS, CreditLifeTable, credit_life_rows

from consolidation.rate_tables import CREDIT_LIFE, PRODUCT11_CARD, CardTable, card_rows

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def card():
    return DecisionTableConfig.load(json.loads((ROOT / "configs" / "0.1.0" / "rate_card_product11.json").read_text()))


def table(rows):
    return param_table(RATE_CARD_COLUMNS, default=rows)


def test_the_table_prices_every_amount_term_and_grade_like_the_index(card):
    rows, version = card_rows(card, PRODUCT11_CARD)
    index, lookup = RateCardIndex.from_card(card), CardTable(table(rows), version, PRODUCT11_CARD)
    amounts, terms = (5_000.0, 10_000.0, 60_000.0, 211_103.11, 499_999.0, 900_000.0), (6, 12, 37, 72, 84, 120)
    for amount, term, grade in itertools.product(amounts, terms, range(0, 14)):
        assert lookup.lookup(amount, term, grade) == index.lookup(amount, term, grade)
    assert lookup.version == index.version == version


def test_a_card_with_cell_ids_off_the_bands_is_refused(card):
    document = json.loads((ROOT / "configs" / "0.1.0" / "rate_card_product11.json").read_text())
    document["rows"][0]["cell_id"] = "rate_card.product11@rc-p11-2026.09#9|9|9"
    with pytest.raises(ValueError, match="cell ids must be"):
        card_rows(DecisionTableConfig.load(document), PRODUCT11_CARD)


def test_the_credit_life_table_rates_like_the_index():
    rows, index, life = credit_life_rows(), CreditLifeIndex(), CreditLifeTable(CREDIT_LIFE, "cl-2026.09")
    assert len(rows) > 3
    for age, term, joint in itertools.product((20.0, 29.9, 30.0, 45.0, 70.0), (12, 24, 36, 60, 84), (False, True)):
        assert life.lookup(age, term, joint) == index.lookup(age, term, joint)
