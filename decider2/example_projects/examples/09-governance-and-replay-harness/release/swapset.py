"""Ordered, attributed swap-set analysis. FRAME TIER, and deliberately so.

The harness has the same two-tier shape as the framework, and it is not a
coincidence - it is the same boundary. Per-decision work (replay, what-if,
explanation) is RECORD tier: `score()`, one record, no polars, microseconds of
execution inside a 5-second budget that is really an I/O budget. Population
work (swap-set, coverage, cohorts, drift) is FRAME tier: 2 M records, joins and
group-bys, 30 minutes.

A swap set is: run generation A and generation B over the same 2 M records with
everything else held identical, join record-by-record on decision_id, and
aggregate. Every line of that sentence is a polars operation except "run", and
"run" is `pipeline.apply(frame, params=...)` - one kernel per module, one
boundary crossing per kernel, which is the regime doc 02 §1.1 measured.

THE COST, BUDGETED HONESTLY (spec §5.5)
  5 increments -> 6 runs over 2 M records at ~30 min each = 2.5 hours per
  release. Acceptable monthly. Unacceptable daily. Spec §11.5's scenario -
  the committee asking for this on weekly releases too - takes it from ~30
  hours a year to ~260, and the answer is not optimisation, it is that a team
  shipping weekly must ship ONE change at a time or accept unattributed swap
  sets, and which they chose is printed in the report.

WHERE THE COMPUTE ACTUALLY GOES, AND THE ONE OPTIMISATION WORTH HAVING
  Increments 1-4 of the worked release (table, param, interior, overlay) do
  not change the structure fingerprint of the flow for 3 of 4 cases, so runs
  1-4 share one compiled image and differ only in the params bundle - doc 08
  §4's `rt.params.swap()` is microseconds and does not recompile. That means
  five of the six runs are the same kernel with a different NamedTuple, which
  is why the 30-minute figure is a data-movement figure and not a compile
  figure. The interior increment stages one background compile; the skeleton
  increment is a different image and the only one that cannot share.
"""

from __future__ import annotations

import polars as pl

from release.increment import Increment, Release


class SwapMeasures:
    """Spec §5.5's required table, per increment AND cumulative."""
    approvals_lost: int
    approvals_gained: int
    net_approval_pp: float
    amount_increased: int
    amount_decreased: int
    net_exposure_move: int                 # cents
    referral_delta: int                    # against a queue sized for 4 000
    reason_code_deltas: dict[int, int]     # code 214 +1 210; code 318 -940
    expected_bad_rate_gained: float        # modelled, against portfolio
    outcome_same_reasons_changed: int      # 3 400 - the row everyone forgets
    uncomparable: int                      # 112
    uncomparable_why: dict[str, int]


def population(flow: str, month: str) -> pl.LazyFrame:
    """Reproducibly, by identifier. Spec §5.5: 'the population used (by
    identifier, reproducibly)'. A month is not a population - a month plus a
    filter plus a random sample is, and the sample must be a deterministic hash
    of application_id and a named seed, never `sample(n)`."""
    pass


def attribute(release: Release, pop: pl.LazyFrame) -> "AttributedSwapSet":
    """n+1 runs, cumulative, in the declared order."""
    pass  # materialise each generation in order; apply over pop; join on decision_id; diff


def complementary(release: Release, pop: pl.LazyFrame) -> "AttributedSwapSet":
    """The reverse order. Run where an interaction is suspected.

    Spec §5.5: 'a difference between the two attributions is itself the
    finding'. In the worked example it is: the rate-card refresh and the tenure
    threshold move attribute 640 vs 210 approvals to increment 2 depending on
    order, because the threshold admits applicants into amount bands whose
    rates the card moved. See artefacts/swapset-REL-2027-04-FLX.md.

    The honest limit: for n increments there are n! orders and we run 2. Two
    orders detect that an interaction exists and bound it; they do not
    decompose it. A Shapley decomposition over 5 increments is 120 runs and
    60 hours of compute, which nobody will pay for monthly. FRAMEWORK-DEMANDS X3.
    """
    pass


def four_causes(flow: str, quarter: str) -> "CauseDecomposition":
    """The committee's standing question (§5.14.4): of this quarter's decline-
    rate movement, how much was the client population, how much the input data,
    how much the logic, and how much what we ourselves approved?

    Population: hold logic, overlays and data-definition fixed; run this
                quarter's applicants through last quarter's generation.
    Data:       same generation, same applicants, but features recomputed under
                the previous feature definitions - only possible where the
                drift monitor's baselines are versioned.
    Logic:      the sum of the non-overlay increments across the quarter's
                releases, already attributed.
    Overlays:   the overlay increments, plus the stack-disabled run
                (adjustments/unwind.py).

    The four do not sum to the total and saying they do would be a lie. The
    residual is reported as `unattributed` and carried forward as an open item
    until it is explained - which is what spec §5.10 requires for any material
    movement anyway.
    """
    pass
