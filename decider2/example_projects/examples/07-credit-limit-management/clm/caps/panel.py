"""The cap panel (s5.5). Seven caps, lowest wins, and the one that bound is
recorded -- always, with every cap's computed value beside it.

WHY THIS IS NOT A WATERFALL. Doc 03 s3.2 would express this as a chain of
modules each overwriting `proposed_limit` with a `min`, and the version chain
would record the sequence. Three requirements break that:

  * s5.5 wants EVERY cap's computed value on the record for all 4.1 M accounts,
    not just the binding one. A waterfall discards a cap's value the moment a
    lower one overwrites it, and the version chain only shows the values that
    actually moved the number.
  * s5.5 wants the binding cap NAMED, and ties broken by a declared order. In a
    waterfall a tie is invisible: the second `min` is a no-op indistinguishable
    from a cap that did not apply.
  * The version chain is a trace-mode artefact. This is a production column.

So a `panel` is a reduction over a named set of independently authored
alternatives, all evaluated, with attribution -- the same mechanism the
exclusion panel and the decrease-trigger panel use, with a different reducer.

Change scenario 7 (a new deposit-balance cap) is then: one module, one list
entry, one code in `vocabulary.CAPS`. It appears in `binding_cap_code`, in the
simulation's cap incidence and in every audit record from that date, because
all three read the panel's declared membership rather than a hand-kept list.
"""

from decider2 import module, panel, param
from clm.caps.exposure_and_income import (AffordabilityCap, GroupExposureCap,
                                          IncomeMultipleCap, TotalUnsecuredCap)
from clm.caps.observed_spend import ObservedSpendCap
from clm.vocabulary import CAPS


def product_maximum_c(product_code: int, tables) -> int:
    """C1 - R300 000 card, R150 000 facility, from the product table."""
    return tables.product_limits.maximum_c[tables.product_limits.cell(product_code)]


def matrix_max_increase_cap_c(current_limit_c: int, matrix_max_increase_c: int) -> int:
    """C6 - the cell's maximum absolute increase, AFTER the overlay stack."""
    return current_limit_c + matrix_max_increase_c


ProductMaximumCap = module(product_maximum_c, name="product_maximum")
MatrixMaxIncreaseCap = module(matrix_max_increase_cap_c, name="matrix_max_increase")

LimitCaps = panel(
    "limit_caps",
    members=[ProductMaximumCap, IncomeMultipleCap, TotalUnsecuredCap,
             GroupExposureCap, ObservedSpendCap, MatrixMaxIncreaseCap,
             AffordabilityCap],
    reduce="min",
    over="matrix_target_limit_c",       # the value being capped; also a member of the min
    codes=CAPS,
    tie_break="declared_order",         # CAPS.tie_order; both ties recorded
    writes={"value": "capped_limit_c",
            "binding": "binding_cap_code",
            "ties": "binding_cap_ties"},
    evidence=["*"],                     # all seven computed values, every account
)


def proposed_limit_c(
    capped_limit_c: int,
    rounding_step_c: int = param(50_000, ge=10_000, le=100_000,
                                 description="R500 in cents; rounding is DOWNWARD"),
) -> int:
    """s5.5 rounding: DOWN to the nearest R500, without exception. A cap that can
    be breached by rounding is not a cap."""
    return (capped_limit_c // rounding_step_c) * rounding_step_c


def additional_limit_c(proposed_limit_c: int, current_limit_c: int) -> int:
    """The increase. Negative or zero means the matrix proposed nothing."""
    return proposed_limit_c - current_limit_c


RoundDown = module(proposed_limit_c, additional_limit_c, name="rounding",
                   evidence=["proposed_limit_c", "binding_cap_code"])

CapChain = LimitCaps | RoundDown
