"""The do-nothing position, and whether the search is warranted at all.

Two jobs, and they are separate on purpose.

1.  THE BASELINE IS THE DENOMINATOR OF EVERY OBJECTIVE MEASURE. search/
    measures.py expresses every component as a ratio to this, which is what
    makes measures absolute and therefore what makes tiered evaluation and
    replay-with-a-bigger-budget safe. If the baseline were wrong, every score
    would be wrong consistently, which is the worst kind of wrong.

2.  The short-circuit decides whether to run the search. It is cheap, it is
    terminal in both directions, and it has to record its reasoning either way.

The baseline is computed ONCE per assessment, on a one-row frame, and then
broadcast into the candidate frame by the cross join in search/plan.py. It is
therefore an `invariant` in the Search declaration, alongside income - and for
the same reason: two scenarios in one assessment that disagree about the
baseline are a defect, not a difference of opinion.
"""

from decider2 import module, param, step
from decider2.values import missing_as

from credit_core import affordability, obligations


@step(output="baseline_total_commitment")
def baseline_total_commitment(existing_obligations: float) -> float:
    """Sum of treated instalments across the FULL inventory, unknowns included.

    Unknown-provider accounts (settleability 7) are excluded from settlement sets
    but REMAIN IN THE OBLIGATION FIGURE. Dropping them here would inflate every
    scenario's measured relief by the amount of debt the Bank could not see,
    which is the most flattering possible error and therefore the one to guard
    against explicitly.
    """
    pass  # core.obligations over the full inventory


@step(output="baseline_weighted_rate")
def baseline_weighted_rate(
    nominal_annual_rate: float,
    balance: float,
    rate_inference_confidence: int,
) -> float:
    """Balance-weighted rate across the inventory, with inference confidence kept.

    Known exactly for internal accounts; reported by the bureau for 38% of
    external accounts; INFERRED from balance, instalment and remaining term
    otherwise. The confidence flag propagates all the way to the comparison,
    because "your current average rate is 24.8%" is a number the client will
    dispute and a third of it is inferred.
    """
    pass  # weighted mean, with a confidence roll-up alongside


@step(output="baseline_total_remaining_cost")
def baseline_total_remaining_cost(
    remaining_term_months: int,
    instalment: float,
    balance: float,
    nominal_annual_rate: float,
    is_revolving: bool,
    paydown_horizon_months: int = param(36, ge=12, le=84),
) -> float:
    """Cost of getting debt-free with no consolidation. The anti-harm denominator.

    Term accounts: remaining instalments x instalment. Revolving accounts: the
    cost of amortising the balance at its current rate over the policy paydown
    horizon.

    REVOLVING DEBT HAS NO NATURAL TERM, so the comparison needs an assumed one,
    and the assumption must be a DECLARED PARAMETER rather than a number in
    someone's head. 36 months is the Bank's; move it to 60 and every balance
    transfer scenario suddenly looks harmless. That sensitivity is the argument
    for the parameter, not against it - it should be visible, owned and dated.
    """
    pass  # term accounts direct; revolving amortised over the horizon


@step(output="baseline_dsr")
def baseline_dsr(existing_obligations: float, net_monthly_income: float) -> float:
    """Debt service ratio before anything happens."""
    pass  # existing_obligations / net_monthly_income


# --- the short-circuit --------------------------------------------------------
#
# Spec 5.4. Consolidation is UNNECESSARY when all six conditions hold, and
# PREFERABLE (and must be evaluated and presented alongside the plain offer) when
# any of five hold.
#
# Every condition is evaluated and every verdict recorded even when the first one
# already settles it. That is deliberately wasteful - six comparisons instead of
# a short-circuit - and it is what spec 5.4's Records paragraph demands: a client
# told "we can only lend you R14 000" who later discovers a consolidation would
# have released R60 000 is a complaint, and the answer must be on file.
#
# Note the asymmetry the spec builds in and this module preserves: the two lists
# are NOT complements. A client can fail "unnecessary" without meeting
# "preferable", in which case the search runs but the plain offer is still shown.
# Encoding them as one boolean would lose that third state.


@step(output="short_circuit_verdict")
def short_circuit_verdict(
    plain_offer_affordable: bool,
    post_advance_dsr: float,
    any_account_in_arrears: bool,
    weighted_rate_gap_bps: float,
    active_account_count: int,
    max_settleable_rate: float,
    max_dsr_pct: float = param(40.0, ge=20.0, le=60.0),
    max_rate_gap_bps: float = param(300.0, ge=0.0, le=1000.0),
    max_active_accounts: int = param(6, ge=2, le=20),
    high_rate_threshold: float = param(28.0, ge=15.0, le=40.0),
) -> int:
    """NOT_NEEDED / PREFERABLE_SHOW_BOTH / SEARCH_REQUIRED, with every condition kept.

    NOT_NEEDED           all six unnecessary-conditions hold. Complete as plain
                         granting under project 03, and record that consolidation
                         was considered and why it was not pursued.
    PREFERABLE_SHOW_BOTH the plain offer passes but at least one preferable-
                         condition holds. Run the search AND present the plain
                         offer beside its results.
    SEARCH_REQUIRED      the plain offer fails affordability. The search is not
                         optional.
    """
    pass  # three-way; each condition's own verdict is a separate tapped output


Baseline = module(
    baseline_total_commitment,
    baseline_weighted_rate,
    baseline_total_remaining_cost,
    baseline_dsr,
    short_circuit_verdict,
    name="baseline",
    taps=["short_circuit_verdict", "baseline_weighted_rate", "baseline_total_remaining_cost"],
    contract="contracts/baseline.json",
)
