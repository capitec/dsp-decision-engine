"""The lookups index a table once per params bundle, and a changed table gives the changed answer."""
from collections import namedtuple

import numpy as np

from loan_granting import rate_tables
from loan_granting.rate_tables import CreditLifeTable, RateCardTable, credit_life_rows, rate_card_rows


def table(rows, writeable=False):
    columns = namedtuple("Table", rows[0])(*(np.array([r[c] for r in rows]) for c in rows[0]))
    for column in columns:
        column.flags.writeable = writeable
    return columns


def test_a_second_lookup_over_the_same_table_reuses_its_index(rate_card):
    rows, version = rate_card_rows(rate_card)
    card = table(rows)
    first, second = RateCardTable(card, version), RateCardTable(card, version)
    first.lookup(50_000.0, 60, 9)
    assert second._grades is first._grades and 9 in second._grades
    assert CreditLifeTable(life := table(credit_life_rows()), "cl")._rows is CreditLifeTable(life, "cl")._rows


def test_a_changed_rate_card_answers_with_its_own_rates(rate_card):
    rows, version = rate_card_rows(rate_card)
    old = RateCardTable(table(rows), version)
    new = RateCardTable(table([{**row, "rate": row["rate"] + 1.0} for row in rows]), version)
    assert new.lookup(50_000.0, 60, 9)[0] == old.lookup(50_000.0, 60, 9)[0] + 1.0
    assert [r for _, r in new.amount_bands(60, 9)] == [r + 1.0 for _, r in old.amount_bands(60, 9)]


def test_a_changed_credit_life_table_answers_with_its_own_rates():
    rows = credit_life_rows()
    old = CreditLifeTable(table(rows), "cl")
    new = CreditLifeTable(table([{**row, "rate": row["rate"] + 1.0} for row in rows]), "cl")
    assert new.lookup(38.0, 60, False)[0] == old.lookup(38.0, 60, False)[0] + 1.0


def test_a_table_whose_arrays_can_change_is_not_remembered(rate_card):
    rows, version = rate_card_rows(rate_card)
    card = table(rows, writeable=True)
    before = RateCardTable(card, version).lookup(50_000.0, 60, 9)[0]
    card.rate[:] += 1.0
    assert RateCardTable(card, version).lookup(50_000.0, 60, 9)[0] == before + 1.0
    assert id(card) not in rate_tables._BUILT
