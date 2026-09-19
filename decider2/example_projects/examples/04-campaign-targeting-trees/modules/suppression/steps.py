"""The suppression predicates that are too specific to be registry rows.

Most of the 34 are registry rows — a feature, an operator and a value.  Five are
table lookups over a key the ruleset algebra cannot express, and those are
registered steps referenced from the registry by id (doc 08 §3.2: a derived value
is a step, not an expression string).
"""

from __future__ import annotations

from decider2 import Table, param, step


@step(description="S21 — client declined this product within the cooling-off period")
def in_cooling_off(
    days_since_last_decline: float | None,
    last_decline_product_code: int | None,
    last_decline_outcome_code: int | None,
    cooling_off: Table = Table("registries/cooling_off.csv",
                               key=("product_code", "prior_outcome_code")),
) -> bool:
    """Cooling-off is a table, not a constant: 8 products x 3 prior outcomes,
    30..180 days.  A Home Loan Further Advance decline suppresses for 180 days;
    an Everyday Card offer not taken suppresses for 45."""
    pass  # null decline date -> not suppressed (never contacted is not a decline)


@step(description="S23 — product already held at or above the offered tier")
def product_held_at_or_above_tier(
    held_tier_code: int | None,
    offer_tier_code: int | None,
) -> bool:
    """2.41 M clients in the September cycle.  Measurement-relevant, so these
    clients still traverse the tree and still get a path: the campaign owner's
    "how big would this campaign be if we could offer it" question is answered
    from their leaves."""
    pass


@step(description="S30/S31 — contact fatigue, all-channel cap and per-channel sub-caps")
def fatigue_exceeded(
    contacts_30d_all: int,
    contacts_30d_sms: int,
    contacts_30d_call: int,
    contacts_30d_email: int,
    contacts_30d_inapp: int,
    cap_all: int = param(4, ge=0, le=20),
    cap_sms: int = param(2, ge=0, le=10),
    cap_call: int = param(1, ge=0, le=5),
    cap_email: int = param(4, ge=0, le=20),
    cap_inapp: int = param(6, ge=0, le=20),
) -> int:
    """Returns a per-channel mask, not a bool.

    The counts arrive from `fatigue.CarriedContactState`, which is a **declared
    carried input** with its own version (see modules/fatigue).  Fatigue is
    evaluated against contacts already made, including those issued by daily
    deltas since the last monthly run, so it is not derivable from the current
    cycle alone — which makes it the one input in this project that a replay
    cannot reconstruct from the feature snapshot.  Hence DEMANDS #30.

    Scenario 8 — the all-channel cap tightening from 4 to 3 programme-wide at two
    weeks' notice — is a single params value with a validator, owned by Channel
    Operations, landing in the params digest.  No tree changes.
    """
    pass


class ScopedSuppressions:
    """Scope is data on the registry row, not a separate mechanism.

        scope: "all"       -> contributes to supp_mask_global
        scope: "campaign"  -> contributes to supp_mask_global, evaluated per campaign
        scope: "channel"   -> contributes to the named channel mask

    The evidence preserves the distinction (spec §5.2 req 1) because the mask a
    bit lives in *is* the scope, and the bit dictionary names it.
    """
