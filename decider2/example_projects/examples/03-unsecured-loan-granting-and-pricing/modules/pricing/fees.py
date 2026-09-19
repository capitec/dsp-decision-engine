"""Statutory fees.  Ten lines of arithmetic with a regulatory finding attached.

The initiation fee is the reason the instalment is not a smooth function of the
amount, quite apart from the rate card:

    R180 plus 10.00% of the advance in excess of R1 000, capped at R1 350,
    both excluding indirect tax at 15%.

The ceiling binds at an advance of R12 700 EXACTLY.  That kink sits INSIDE the
rate band R12 000-R12 999, so within a single rate band the instalment's
dependence on the amount changes shape.  A search that partitions its domain on
rate-band edges alone is therefore wrong -- and wrong in exactly the silent way
§5.8 warns about.

`@breakpoints` is how the fee schedule tells the search where its kinks are.
The decorator appears three times in this project -- here, on the rate card
axis, and on the credit-life term bands -- and the search's partition is the
UNION of everything so declared.  Change scenario 10 moves this kink from
R12 700 to R13 300 and lands it in a different band: the declaration moves with
the parameter and the search stays correct with no code change.
See FRAMEWORK-DEMANDS #11.
"""

from __future__ import annotations

from decider2 import breakpoints, module, monotone_in, param, step
from decider2.money import Money


@breakpoints("offered_amount", at=["initiation_fee_ceiling_binds_at"])
@monotone_in("offered_amount", direction="non_decreasing")
@step(output="initiation_fee")
def initiation_fee(
    offered_amount: Money,
    tables,
    base: Money = param(Money("180.00"), owner="compliance"),
    marginal_rate: float = param(0.10, ge=0, le=1, owner="compliance"),
    ceiling: Money = param(Money("1350.00"), owner="compliance"),
    threshold: Money = param(Money("1000.00"), owner="compliance"),
) -> Money:
    """Base plus a marginal rate on the excess, subject to a statutory ceiling.

    The four params are Compliance-owned and annually adjusted.  They are
    params rather than a table because there are four of them and they have no
    key -- but they carry `owner=`, which is what stops a product manager
    moving the statutory fee cap (§6.2's three ownership classes).
    See FRAMEWORK-DEMANDS #15.
    """
    pass


@step(output="initiation_fee_ceiling_binds_at")
def ceiling_binds_at(ceiling: Money, base: Money, marginal_rate: float,
                     threshold: Money) -> Money:
    """The advance at which the fee ceiling binds. R12 700.00 at today's values.

    Derived, not written down.  If it were a literal it would be a second
    canonical location for the same fact (doc 01 §5.3), and change scenario 10
    would need two edits, one of which somebody would miss.
    """
    pass


@step(output="amount_financed")
def amount_financed(offered_amount: Money, initiation_fee_incl_tax: Money) -> Money:
    """The advance plus the CAPITALISED initiation fee.

    `offered_amount` and `amount_financed` are different numbers and the
    disclosure names both.  Naming them `amount` and `amount2` -- or worse,
    reusing one name -- is how a reconciliation break is born, so the project
    vocabulary (doc 03 §5.2) pins both names and a lint rejects a step that
    takes one and returns the other under the first name.
    """
    pass


@step(output="monthly_service_fee")
def monthly_service_fee(
    tables,
    fee_excl_tax: Money = param(Money("72.00"), owner="compliance"),
    tax_rate: float = param(0.15, ge=0, le=1, owner="compliance"),
) -> Money:
    """R72.00 excluding tax, R82.80 including. Statutorily capped, effective-dated."""
    pass


Fees = module(initiation_fee, ceiling_binds_at, amount_financed, monthly_service_fee,
              name="fees")
