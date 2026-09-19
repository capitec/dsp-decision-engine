"""Declared overlay application sites.

An overlay changes an answer.  There are exactly three places in this pipeline
where an overlay may do so, and all three are **modules in the skeleton**, present
whether or not any overlay is in force:

    1.  `ScoreShift`     — an input transform, before the tree.  Score shifts and
                           odds multipliers land here.
    2.  the thresholds array of `CampaignTreeStage` — cut-off shifts and volume
                           dials land here, as substituted array elements.
    3.  `AmountCapOverlay` — after pre-assessment, before the leaf's amount rule.
                           Cap reductions land here.

Declaring the sites in code is what makes "applying an overlay outside its
declared scope is an error that fails the cycle, not a silent no-op" (§5.3.4 req 6)
true for *kinds* as well as for scopes.  An overlay of a kind with no site cannot
be applied, and `resolve()` says so by name instead of quietly doing nothing.

Doc 03 has no notion of a declared site; an overlay would be "some params that
some step happens to read", and an overlay kind nobody wired would be invisible.
DEMANDS #25.
"""

from __future__ import annotations

from decider2 import module, param, step


@step(description="Overlay site 1 — score shift and odds multiplier")
def shifted_behaviour_score(
    behaviour_score: float | None,
    score_shift: float = param(0.0, ge=-100.0, le=100.0,
                               description="Points added. Valued from the overlay stack; "
                                           "0.0 when no score overlay is in force."),
) -> float | None:
    """`behaviour_score_unadjusted` survives as its own value and is what the
    unadjusted evaluation reads (00 §6.22 property 2)."""
    pass


@step(description="Overlay site 1b — PD multiplier used in arbitration expected value")
def adjusted_pd(
    probability_of_default: float,
    odds_multiplier: float = param(1.0, ge=0.25, le=4.0),
) -> float:
    pass


@step(description="Overlay site 3 — amount cap reduction. MAY ONLY REDUCE.")
def overlaid_amount_cap_cents(
    pre_assessed_amount_cents: int,
    tier_cap_cents: int,
    cap_reduction_pct: float = param(0.0, ge=0.0, le=0.9,
                                     description="Fraction removed from the tier ceiling."),
) -> int:
    """Spec §5.5 req 6: an overlay that would raise an advertised amount above the
    pre-assessed amount is prohibited **by construction**, not by policy.  The
    validator's bound is `ge=0.0`, the step takes `min(...)` against the
    pre-assessed amount, and there is no uplift dial to misconfigure.  An uplift
    dial is a direct attack on "nothing is advertised the Bank would not grant",
    so the way to make it impossible is to have no field that could express it.
    """
    pass


ScoreShift = module(shifted_behaviour_score, adjusted_pd,
                    name="overlay_score_shift", partition="record",
                    taps=["score_shift", "odds_multiplier"])

AmountCapOverlay = module(overlaid_amount_cap_cents,
                          name="overlay_amount_cap", partition="record",
                          taps=["cap_reduction_pct"])
