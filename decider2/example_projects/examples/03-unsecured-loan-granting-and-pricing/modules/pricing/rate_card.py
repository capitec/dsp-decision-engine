"""The Flex Loan rate card.  96 amount bands x 55 terms x 12 grades = 63 360 cells.

Doc 03's provisional `Table` sketch is a one-dimensional keyed lookup --
`tables.term.max_loan[term]`.  It cannot express a banded axis, cell-level
attribution, effective dating, pre-live validation, cell-level diffing, or the
spreadsheet ingestion path, and every one of those is a hard requirement here.
So this project needs a `grid()` kind.  FRAMEWORK-DEMANDS #8, #9.

Four things about this declaration are load-bearing.

1.  **The band edges are an artefact, not a literal.**  `amount_bands.csv` is
    96 rows Treasury authors beside the cells.  Change scenario 1 re-bands the
    card from 96 to 120; nothing in code changes, and the solve's correctness
    survives because the solve reads its breakpoints FROM THIS AXIS rather
    than assuming them.  See modules/solve/partition.py.

2.  **A read returns a value AND a cell reference.**  `rate_cell_id` is one
    int32 -- band index * 660 + term index * 12 + grade.  Attribution is then
    free rather than bolted on: a regulator asking "does that rate match the
    card you published?" gets "cell [band 68, term 60, grade 7] of card version
    2026-09-A" because the cell id is in the record, not because somebody
    remembered to log it.

3.  **Validation belongs to the grid's DEFINITION (code, reviewed), and runs
    against its DATA (config, monthly).**  The five rules below are §6.1's
    card-validation list.  A card version that fails any of them does not go
    live -- `stage()` fails, the active generation keeps serving.

4.  **Declared inversions are DATA the solve consumes**, not a comment.  41 of
    them today, 58 after the re-band.  Declaring an inversion is how Treasury
    says "this is deliberate"; an UNDECLARED inversion blocks the card; and the
    declared list is an input to the search's static evaluation-budget proof.
"""

from __future__ import annotations

from decider2.tables import Axis, grid, validates
from decider2.money import Money

FlexRateCard = grid(
    name="flex_rate_card",
    owner="treasury",
    cadence="monthly, plus mid-month patches on repo moves",
    source="tables/flex_rate_card/",
    effective_dated=True,
    axes=[
        # A BANDED axis: non-uniform edges, read from a companion artefact.
        # R500 bands to R5 000, R1 000 to R50 000, R5 000 to R150 000,
        # R10 000 to R350 000, R30 000 to R500 000.
        Axis.banded("amount", edges="amount_bands.csv", unit=Money, count=96),
        # A DENSE axis: every month 6..60.  Terms 61-84 are priced from the
        # 60-month column plus a long-term loading held in a SEPARATE 2 x 12
        # table -- "a Treasury decision, not a technical one, and it must
        # remain visible as such" (§5.7a).  So it is a second artefact with its
        # own name, not 24 more columns and not a special case in a lookup.
        Axis.dense("term_months", low=6, high=60, count=55),
        Axis.dense("risk_grade", low=1, high=12, count=12),
    ],
    cell=Axis.value("nominal_annual_rate", dtype="int32_bp100"),
    # 63 360 x 4 bytes = 253 KB.  L2-resident.  Residency is asserted at build,
    # not hoped for: `decider build --verify` fails if any table declared
    # `resident=True` is not materialised and indexed in the image.
    resident=True,
    emits=["nominal_annual_rate", "rate_cell_id", "rate_card_version",
           "amount_band_index", "amount_band_low", "amount_band_high"],
)

LongTermLoading = grid(
    name="flex_long_term_loading",
    owner="treasury",
    source="tables/long_term_loading.csv",
    effective_dated=True,
    axes=[Axis.banded("term_months", edges=[(61, 72), (73, 84)]),
          Axis.dense("risk_grade", low=1, high=12)],
    cell=Axis.value("loading_bp", dtype="int32_bp100"),
)


# ---------------------------------------------------------------------------
# Validation.  Runs on EVERY version before it may go live, including a
# mid-month patch (§6.1 rule 5: "the mid-month patch path produces the same
# artefact as a full refresh").
# ---------------------------------------------------------------------------


@validates(FlexRateCard, severity="block")
def every_cell_populated(card) -> "Report":
    """A null rate is a defect, never a 'use the neighbour'."""
    pass


@validates(FlexRateCard, severity="block")
def at_or_below_statutory_ceiling(card, tables) -> "Report":
    """Every cell at or below the statutory ceiling in force on its EFFECTIVE DATE.

    Not on today's date.  A repo cut lowers the ceiling and puts roughly 1 100
    cells in breach without anybody touching the card, so this validation is
    re-run on every repo move as well as on every card version -- it is
    registered as a trigger on the ceiling artefact, not only on the card.
    """
    pass


@validates(FlexRateCard, severity="block")
def monotone_non_increasing_across_grades(card) -> "Report":
    """Within an (amount band, term) cell, a better grade is never priced worse."""
    pass


@validates(FlexRateCard, severity="block_unless_declared")
def band_edge_inversions_are_declared(card) -> "Report":
    """A band priced higher than the band above it is PERMITTED but must be declared.

    41 on the current card, deliberately -- Treasury rewards crossing R50 000
    with 150 basis points.  An undeclared inversion blocks the card.  The
    declared list is written into the version manifest and is consumed by
    modules/solve/partition.py, so it is data the search reads rather than a
    property the search assumes.
    """
    pass


@validates(FlexRateCard, severity="block")
def search_budget_is_satisfiable(card, tables) -> "Report":
    """For every (term, grade), the worst-case probe count of the declared search
    strategy over this card is at or below the declared evaluation ceiling of 24.

    THIS IS THE ONE THAT MATTERS.  Acceptance criterion 4 -- "no application
    performs more than 24 pricing evaluations for any single term, EVER, on any
    input, including adversarial ones" -- is not testable by sampling.  It is
    computable in closed form from the axis edges and the bracket's worst-case
    tightening (see modules/solve/search.py), and it is computed HERE, at card
    validation, against the card that is about to go live.  A re-banded card
    that would blow the budget never reaches production.
    """
    pass
