"""The assembled account state — and the contract between batch and real time.

`AccountState` is a declared, versioned, named record type. It is the ONLY thing
downstream of here reads. Batch materialises 2.3M of them at 04:20; the live-call
path materialises exactly one at 10:14:33. Nothing downstream can tell which.

This is the single mechanism behind spec §13.12 ("how do batch and real-time
share one implementation") and NFR "batch/real-time agreement". Doc 02 §3.5 gives
`apply()` and `score()` over the same kernel, which is necessary and not
sufficient: the kernel agrees, but the two paths would still assemble their
inputs differently, and that is where drift actually lives.

So the state vector gets a `@state_vector` declaration with an `incremental=`
clause per field, and the framework REFUSES TO BUILD a pipeline declared
`realtime=True` whose state vector contains a field with no incremental form.
`count over rolling(7d)` has one — add the new events, drop the expired. A p95
over 90 days does not, and if somebody adds one, the build fails naming the
field, at 09:00 on a Tuesday, instead of the nightly reconciliation finding a
material difference six weeks later.

DEVIATION FROM DOC 03: doc 03 has no concept of a named record type spanning a
pipeline. It has leaf inputs, wired by name, inferred per module. That is right
for a 15-input affordability module and wrong for a 190-field state vector read
by nine downstream modules, because the interface of every one of those modules
then lists 40 leaf inputs and a reviewer cannot see the boundary. See
FRAMEWORK-DEMANDS #4.
"""

import polars as pl

from decider2 import module, state_vector, step
from decider2.frame import Join
from decider2.types import Date, Timestamp, cents, i1, i2, i4, i8

from ..timelines import windows as W
from ..timelines.calendar import age_days, bdays_between
from .episode import EpisodeFields


@state_vector(name="account_state", version=7, realtime=True)
class AccountState:
    """Everything §5.1 requires, with every temporal quantity resolved against
    `decision_date` and every window carrying its feed watermark."""

    # --- identity -----------------------------------------------------------
    account_id: i8
    client_id: i8
    product_code: i2
    product_family_code: i1
    decision_date: Date

    # --- position -----------------------------------------------------------
    days_past_due: i2
    arrears_amount: cents
    outstanding_balance: cents
    credit_limit: cents | None          # revolving only
    instalment: cents | None            # null for 41 000 permanent over-limit revolvers

    # --- windows (all from timelines/windows.py; nothing re-derived here) ----
    last_payment_on: Date | None = W.last_payment_on
    paid_30d: cents = W.paid_30d
    paid_90d: cents = W.paid_90d
    payment_pattern_12m: i4 = W.payment_pattern_12m
    times_cured_12m: i2 = W.times_cured_12m
    times_cured_24m: i2 = W.times_cured_24m
    episodes_12m: i2 = W.episodes_12m
    rpc_90d: i2 = W.rpc_90d
    attempts_90d: i2 = W.attempts_90d
    outcome_unknown_90d: i2 = W.outcome_unknown_90d
    untouched_days: i4 = W.untouched_days
    broken_promises_90d: i2 = W.broken_promises_90d
    consecutive_broken: i2 = W.consecutive_broken
    max_intensity_this_episode: i2 = W.max_intensity_this_episode
    automated_only_streak_days: i4 = W.automated_only_streak_days

    # --- client-grain windows, explicitly broadcast -------------------------
    # The framework will not silently read a client-grain window into an
    # account-grain state vector. `broadcast()` is the declaration that says
    # "yes, all of this client's accounts share this number", which is exactly
    # what the per-client frequency cap means and exactly what three accounts
    # each observing a per-account cap gets wrong.
    contacts_7d_client: i2 = W.contacts_7d_client.broadcast(from_grain="client_id")
    spoken_today_client: i2 = W.spoken_today_client.broadcast(from_grain="client_id")
    contacts_today_client: i2 = W.contacts_today_client.broadcast(from_grain="client_id")
    contacts_30d_client: i2 = W.contacts_30d_client.broadcast(from_grain="client_id")

    # --- episode ------------------------------------------------------------
    episode: EpisodeFields

    # --- provenance: every window carries the watermark that produced it -----
    status_feed_watermark: Timestamp = W.status_feed_watermark
    contact_feed_watermark: Timestamp = W.contact_feed_watermark
    payment_feed_watermark: Timestamp = W.payment_feed_watermark
    knowledge_cutoff: Timestamp         # from shared; the as-known / as-at-now switch


# --------------------------------------------------------------------------
# Bucket is COMPUTED, not read. Spec §4.3: "an account under a performing
# arrangement is held at the bucket it entered in; an account cured twice in six
# months is floored at bucket 3." Both are Collections Strategy policy. A bucket
# read from the account master would be neither.
# --------------------------------------------------------------------------

def bucket_from_dpd(days_past_due: i2, edges) -> i1:
    pass  # band lookup against the (parameterised) bucket edges — see matrix/dimensions.py


def bucket_held_by_arrangement(
    bucket_from_dpd: i1,
    arrangement_entry_bucket: i1 | None,
    arrangement_is_performing: bool,
) -> i1:
    pass  # hold at entry bucket while the arrangement performs; else fall through


@step(output="arrears_bucket_code")
def bucket_floored_by_cure_history(
    bucket_held_by_arrangement: i1,
    times_cured_12m: i2,
    floor_bucket: i1 = param(3, ge=1, le=8),
    floor_after_cures: i2 = param(2, ge=1, le=6),
) -> i1:
    """Floor the treatment bucket for repeat curers. Collections Strategy owns both knobs."""
    pass  # max(bucket, floor_bucket) when times_cured_12m >= floor_after_cures


BucketAssignment = module(
    bucket_from_dpd,
    bucket_held_by_arrangement,
    bucket_floored_by_cure_history,
    name="bucket_assignment",
    taps=["bucket_from_dpd", "bucket_held_by_arrangement", "arrears_bucket_code"],
    # three taps, three versions of the same idea, so the audit answer to "why is
    # this account in bucket 3 with 200 days past due" is one column read.
)


# --------------------------------------------------------------------------
# Assembly is frame tier. It is the ONE mandatory boundary crossing in the flow:
# 180M contact rows, 27.6M payment rows and 55.2M roll rows reduce to ~190
# scalars per account, once. Everything after this is record tier.
# --------------------------------------------------------------------------

AssembleState = (
    AccountState.assemble()          # emits one frame-tier pass per timeline
    | BucketAssignment
)


# The bureau staleness trichotomy the spec insists on (§4.2): "no bureau view",
# "bureau view up to 13 months old" and "fresh" are three states, not two.
# Expressed as a declared fill policy plus an explicit age, never as a null test
# in forty downstream bodies.
def bureau_state_code(
    bureau_as_of_date: Date | None,
    decision_date: Date,
    stale_after_days: i4 = param(45, ge=1, le=400),
) -> i1:
    """0 = none held, 1 = fresh, 2 = carried (stale but present)."""
    pass
