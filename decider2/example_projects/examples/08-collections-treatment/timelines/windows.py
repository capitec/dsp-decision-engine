"""Every time window in the project, declared as data.

This file is the answer to spec §13.10 — "how are time windows expressed so that
30/60/90-day, 12/24-month, rolling-7-day, business-day and until-a-date windows
are one concept rather than forty hand-written filters".

They are one concept. It is `window=`. There are 41 of them here and they are
the entire temporal surface of the project: nothing downstream writes a date
comparison against a raw event table, and nothing anywhere calls now().

MECHANICS. Each declaration below is a Window: a (timeline, predicate,
aggregate, bound) tuple. `decider2` groups every Window over the same timeline
into ONE frame-tier pass — a single scan of the 180M-row contact log producing
23 columns, not 23 scans. The record tier then sees 23 plain scalars. The author
never writes the join, and `lineage("treatment_code")` still resolves through
each window back to the timeline and the predicate that selected its rows.

BOUNDS. `rolling(days=7)` ends at `decision_date` implicitly. You may only end
a window somewhere else by saying so — `rolling(days=7, ending=promise_date)`.
This default is deliberate and it is the difference between a replayable project
and one that quietly re-decides the past. Passing a literal or a wall clock to
`ending=` is a build error.
"""

from decider2.temporal import (
    Window, rolling, months, since, until, business_days, at_most_one, latest, earliest,
)
from decider2.types import Date, cents, i2, i4

from .streams import (
    ARRANGEMENT_INSTALMENTS, BUCKET_ROLLS, CONTACTS_BY_ACCOUNT, CONTACTS_BY_CLIENT,
    PAYMENTS, PROMISES, STATUS_EVENTS, TREATMENTS,
)
from . import schemas as S

Ev = Window.fields   # typed field accessor; `Ev.channel_code == 1` is data, not a lambda

# ============================================================ payments =======

last_payment_on: Date | None = PAYMENTS.agg(
    latest(Ev.value_date), where=~Ev.is_reversal, window=rolling(days=800))
last_payment_amount: cents = PAYMENTS.agg(
    latest(Ev.amount), where=~Ev.is_reversal, window=rolling(days=800), default=0)
paid_30d: cents = PAYMENTS.agg(sum(Ev.amount), where=~Ev.is_reversal, window=rolling(days=30))
paid_60d: cents = PAYMENTS.agg(sum(Ev.amount), where=~Ev.is_reversal, window=rolling(days=60))
paid_90d: cents = PAYMENTS.agg(sum(Ev.amount), where=~Ev.is_reversal, window=rolling(days=90))

# The 12-month paid/partial/missed pattern, as a packed i4 bitfield (2 bits per
# month, 12 months). One column, not twelve, and it rides in a numba register.
payment_pattern_12m: i4 = PAYMENTS.agg(
    Window.pack_monthly(Ev.amount, buckets=3), window=months(12))

# ============================================================ contacts ======
# NOTE THE GRAIN. Frequency caps are per CLIENT; sequence counters are per
# ACCOUNT. Same event table, two timelines, and the compiler will not let you
# read a client-grain window into an account-grain step without an explicit
# broadcast (see state/assembly.py).

contacts_today_client: i2 = CONTACTS_BY_CLIENT.count(window=rolling(days=1))
spoken_today_client: i2 = CONTACTS_BY_CLIENT.count(where=Ev.is_spoken, window=rolling(days=1))
contacts_7d_client: i2 = CONTACTS_BY_CLIENT.count(window=rolling(days=7))
contacts_30d_client: i2 = CONTACTS_BY_CLIENT.count(window=rolling(days=30))

sms_attempts_90d: i2 = CONTACTS_BY_ACCOUNT.count(
    where=Ev.channel_code == 1, window=rolling(days=90))
last_sms_on: Date | None = CONTACTS_BY_ACCOUNT.agg(
    latest(Ev.attempted_on), where=Ev.channel_code == 1, window=rolling(days=90))
last_agent_call_on: Date | None = CONTACTS_BY_ACCOUNT.agg(
    latest(Ev.attempted_on), where=Ev.channel_code == 5, window=rolling(days=90))
last_field_visit_on: Date | None = CONTACTS_BY_ACCOUNT.agg(
    latest(Ev.attempted_on), where=Ev.channel_code == 6, window=rolling(days=365))

# "no answer" and "wrong number" both look like failure and must not be treated
# alike (spec §5.1). Three wrong numbers invalidate the contact point; they do
# not escalate the account. Two windows, one predicate apart.
no_answer_90d: i2 = CONTACTS_BY_ACCOUNT.count(
    where=Ev.outcome_code == 1, window=rolling(days=90))
wrong_number_by_channel_90d: i2 = CONTACTS_BY_ACCOUNT.count(
    where=Ev.outcome_code == 3, window=rolling(days=90), by=Ev.channel_code)
rpc_90d: i2 = CONTACTS_BY_ACCOUNT.count(
    where=Ev.outcome_code.is_in([4, 5, 6, 7]), window=rolling(days=90))
attempts_90d: i2 = CONTACTS_BY_ACCOUNT.count(window=rolling(days=90))
untouched_days: i4 = CONTACTS_BY_ACCOUNT.agg(
    Window.days_since(latest(Ev.attempted_on)), window=rolling(days=400), default=400)

# 4.1% of dialler records have a null outcome. That is a THIRD state, not a
# failure and not a success, and it has its own window so no downstream rule can
# fold it into either by accident.
outcome_unknown_90d: i2 = CONTACTS_BY_ACCOUNT.count(
    where=Ev.outcome_code.is_null(), window=rolling(days=90))

# ============================================================ promises ======

promises_this_episode: i2 = PROMISES.count(window=since("episode_opened_on"))
broken_promises_90d: i2 = PROMISES.count(where=Ev.outcome_code == 3, window=rolling(days=90))
kept_promises_lifetime: i2 = PROMISES.count(where=Ev.outcome_code == 1, window=rolling(days=2600))
consecutive_broken: i2 = PROMISES.agg(
    Window.run_length(Ev.outcome_code == 3, from_end=True), window=rolling(days=400))
open_promise_date: Date | None = PROMISES.agg(
    earliest(Ev.promised_on), where=Ev.outcome_code.is_null(), window=until(days=21))
open_promise_id: i2 | None = PROMISES.agg(
    at_most_one(Ev.promise_id), where=Ev.outcome_code.is_null(), window=until(days=21))

# The promise grace end is BUSINESS days, and it is a per-row date computed from
# another per-row date. It is a step, not a window — see path/sequence.py.

# ========================================================= arrangements =====

arrangements_24m: i2 = ARRANGEMENT_INSTALMENTS.agg(
    Window.distinct_count(Ev.arrangement_id), window=months(24))
consecutive_failed_arrangements: i2 = ARRANGEMENT_INSTALMENTS.agg(
    Window.run_length(Ev.settled_amount < Ev.due_amount, from_end=True), window=months(24))
last_failed_arrangement_on: Date | None = ARRANGEMENT_INSTALMENTS.agg(
    latest(Ev.due_on), where=Ev.settled_amount * 100 < Ev.due_amount * 95, window=months(24))
missed_instalments_current_arrangement: i2 = ARRANGEMENT_INSTALMENTS.count(
    where=Ev.settled_at.is_null(), window=rolling(days=400), by=Ev.arrangement_id)

# ============================================================== rolls =======

times_cured_12m: i2 = BUCKET_ROLLS.agg(
    Window.transitions_to(Ev.arrears_bucket_code, value=0), window=months(12))
times_cured_24m: i2 = BUCKET_ROLLS.agg(
    Window.transitions_to(Ev.arrears_bucket_code, value=0), window=months(24))
worst_bucket_12m: i2 = BUCKET_ROLLS.agg(max(Ev.arrears_bucket_code), window=months(12))
episodes_12m: i2 = BUCKET_ROLLS.agg(
    Window.transitions_from(Ev.arrears_bucket_code, value=0), window=months(12))
bucket_yesterday: i2 | None = BUCKET_ROLLS.agg(
    latest(Ev.arrears_bucket_code), window=rolling(days=2))

# ======================================================= own treatments =====
# Yesterday's decision is today's input. Not a log written afterwards.

treatments_14d = TREATMENTS.rows(window=rolling(days=14), order_by=Ev.decision_date)
last_treatment_code: i2 | None = TREATMENTS.agg(
    latest(Ev.treatment_code), where=Ev.was_dispatched, window=rolling(days=90))
last_treatment_on: Date | None = TREATMENTS.agg(
    latest(Ev.decision_date), where=Ev.was_dispatched, window=rolling(days=90))
max_intensity_this_episode: i2 = TREATMENTS.agg(
    max(Ev.treatment_intensity), where=Ev.was_dispatched, window=since("episode_opened_on"),
    default=0)
attempts_at_current_position: i2 = TREATMENTS.count(
    where=Ev.was_dispatched & (Ev.path_position_after == Window.carried("path_position_in")),
    window=since("episode_opened_on"))
automated_only_streak_days: i4 = TREATMENTS.agg(
    Window.run_length_days(Ev.treatment_code.is_in([1, 2, 3, 4]), from_end=True),
    window=rolling(days=90))

# ============================================================= statuses =====
# Status windows are CLIENT-grain and they carry their source reference through,
# because suspensions/ must name the record that caused each suspension and not
# merely that one existed.

debt_review_stage: i2 | None = STATUS_EVENTS.agg(
    latest(Ev.stage_code), where=Ev.status_code.between(101, 104) & Ev.cleared_on.is_null(),
    window=rolling(days=4000), carry=[Ev.source_reference, Ev.effective_on, Ev.received_at])
open_dispute_ref: str | None = STATUS_EVENTS.agg(
    latest(Ev.source_reference), where=Ev.status_code == 110 & Ev.cleared_on.is_null(),
    window=rolling(days=800), carry=[Ev.effective_on, Ev.received_at])
notice_delivered_on: Date | None = STATUS_EVENTS.agg(
    latest(Ev.effective_on), where=Ev.status_code == 112, window=rolling(days=400),
    carry=[Ev.source_reference, Ev.received_at])
last_acknowledgement_on: Date | None = STATUS_EVENTS.agg(
    latest(Ev.effective_on), where=Ev.status_code == 140, window=rolling(days=4000))
last_legal_process_served_on: Date | None = STATUS_EVENTS.agg(
    latest(Ev.effective_on), where=Ev.status_code == 141, window=rolling(days=4000))

# The feed watermark. Not a window over events — a property of the FEED, and the
# thing §9.2 turns on: "if the status arrived after the run, that must be
# demonstrable from the feed watermark rather than asserted."
status_feed_watermark = STATUS_EVENTS.watermark()
contact_feed_watermark = CONTACTS_BY_ACCOUNT.watermark()
payment_feed_watermark = PAYMENTS.watermark()
