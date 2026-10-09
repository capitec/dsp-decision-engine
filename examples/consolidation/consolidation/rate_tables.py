"""The consolidation loan's rate card, the Flex Loan card and the credit life rates as `param_table`s.

`loan_granting.rate_tables` reads the Flex Loan card; product 11 has its own card with its own cell ids, so its rows
come from `card_rows` and its lookups from `CardTable`. Both read the arrays of a table param, so the data is in the
params document and the diff, not in an object a step closes over.
"""
from __future__ import annotations

import polars as pl
from credit_core.credit_life import CREDIT_LIFE_VERSION
from credit_core.evidence import cell_id
from decider import DecisionTableConfig, param, param_table
from loan_granting.rate_tables import (
    CREDIT_LIFE_COLUMNS,
    OPEN,
    RATE_CARD_COLUMNS,
    RATE_CARD_TABLE,
    CreditLifeTable,
    RateCardTable,
    credit_life_rows,
)

PRODUCT11_CARD = "rate_card.product11"

# One table for every step that prices, so the rates are edited in one place.
CREDIT_LIFE = param_table(CREDIT_LIFE_COLUMNS, shared_key="credit_life", default=credit_life_rows())
CREDIT_LIFE_VERSION_PARAM = param(CREDIT_LIFE_VERSION, shared_key="credit_life_version")


def card_rows(card: DecisionTableConfig, table: str) -> tuple[list[dict], str]:
    """The card's rows sorted by grade, amount band and term band, and the card's version.

    The bands are numbered by the order of their lower edges, and the cell ids must be
    `<table>@<version>#<amt_band>|<term_band>|<grade>`: a card whose ids are anything else is refused.
    """
    frame = card.rows.df.sort("grade", "amt_lo", "term_lo").with_columns(
        amt_band=pl.col("amt_lo").rank("dense").cast(pl.Int64) - 1, term_band=pl.col("term_lo").rank("dense").cast(pl.Int64) - 1)
    versions = frame["rate_card_version"].unique().to_list()
    if len(versions) != 1:
        raise ValueError(f"the rate card holds {len(versions)} versions; a card has exactly one")
    expected = [cell_id(table, versions[0], a, t, g) for a, t, g in zip(*frame.select("amt_band", "term_band", "grade"))]
    if frame["cell_id"].to_list() != expected:
        raise ValueError(f"rate card cell ids must be {table}@<version>#<amt_band>|<term_band>|<grade>")
    return frame.select(*RATE_CARD_COLUMNS).with_columns(
        [frame[c].clip(-OPEN, OPEN) for c in ("amt_lo", "term_lo")]).to_dicts(), versions[0]


class CardTable(RateCardTable):
    """`RateCardTable` for a card whose cell ids start with `table` instead of the Flex Loan's."""

    def __init__(self, card, version: str, table: str):
        super().__init__(card, version)
        self._table = table

    def lookup(self, amount: float, term_months: float, grade: int) -> tuple[float, str] | None:
        hit = super().lookup(amount, term_months, grade)
        return hit and (hit[0], hit[1].replace(RATE_CARD_TABLE, self._table, 1))


def product11_params(card: DecisionTableConfig):
    """`(rows param, version param)` for product 11's card: the card document is their default, shared by the steps that price."""
    rows, version = card_rows(card, PRODUCT11_CARD)
    return (param_table(RATE_CARD_COLUMNS, shared_key="rate_card_product11", default=rows),
            param(version, shared_key="rate_card_product11_version"))


__all__ = ["CREDIT_LIFE", "CREDIT_LIFE_VERSION_PARAM", "PRODUCT11_CARD", "CardTable", "CreditLifeTable", "card_rows", "product11_params"]
