"""Credit life premium.  14 age bands x 8 term bands x 6 employment types = 672 cells.

Two things make this more than a lookup.

1.  **The premium is charged on the amount FINANCED**, which includes the
    capitalised initiation fee, which is a piecewise function of the advance.
    So the premium inherits the fee's kink.  Its own term-band edges are
    irrelevant within one term (the term is fixed during a search) but its
    dependence on `amount_financed` is not, which is why it declares
    monotonicity in the search variable.

2.  **The client may substitute their own policy.**  A statutory right.  When
    `credit_life_substitution_declared` is true the premium is zero, the
    instalment drops, the offer carries a condition that proof of cover is
    produced before disbursement, and -- the part that is easy to get wrong --
    the AFFORDABILITY TEST IS PERFORMED AGAINST THE LOWER INSTALMENT, and the
    record must state that it was.  That makes substitution an input to the
    search, not a post-hoc adjustment to its answer.
"""

from __future__ import annotations

from decider2 import Branch, breakpoints, module, monotone_in, step
from decider2.money import Money
from decider2.tables import Axis, grid, validates

CreditLifeRates = grid(
    name="credit_life_rates",
    owner="credit_risk_policy",
    source="tables/credit_life_rates.csv",
    effective_dated=True,
    axes=[
        Axis.banded("applicant_age_years", edges="age_bands.csv", count=14),
        Axis.banded("term_months", edges=[(6, 12), (13, 18), (19, 24), (25, 36),
                                          (37, 48), (49, 60), (61, 72), (73, 84)]),
        Axis.dense("employment_type_code", low=1, high=6),
    ],
    cell=Axis.value("premium_per_1000", dtype="int64_cents"),
    emits=["credit_life_rate_cell_id", "credit_life_table_version"],
    # Change scenario 8 takes this to 16 x 8 x 7 = 896 cells.  An axis change
    # is a STAGED change, not a free one: the cell array's shape is part of the
    # table's type.  Cell VALUES are free.  Doc 08 §4.2 does not draw that line
    # and it needs to -- FRAMEWORK-DEMANDS #17.
)


@validates(CreditLifeRates, severity="block")
def at_or_below_statutory_cap(card, tables) -> "Report":
    """R4.50 per R1 000 is the statutory ceiling for unsecured credit.
    Any cell exceeding it is a CARD DEFECT, not a price."""
    pass


@breakpoints("offered_amount", inherits_from="fees.initiation_fee")
@monotone_in("offered_amount", direction="non_decreasing")
@step(output="credit_life_premium")
def credit_life_premium(
    amount_financed: Money,
    applicant_age_years: float,
    term_months: int,
    employment_type_code: int,
    credit_life_substitution_declared: bool,
    tables,
) -> Money:
    """Monthly premium, per R1 000 of the amount financed. Zero on substitution."""
    pass


@step(output="offer_conditions")
def substitution_condition(credit_life_substitution_declared: bool) -> int:
    """Proof of cover before disbursement, where the client substituted."""
    pass


CreditLife = module(credit_life_premium, substitution_condition, name="credit_life",
                    taps=["credit_life_rate_cell_id", "credit_life_substitution_declared"])
