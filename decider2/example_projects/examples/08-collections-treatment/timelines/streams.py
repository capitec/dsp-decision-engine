"""Event streams, declared once.

A `Timeline` is a declared, versioned event source. It is NOT a polars frame and
it is NOT a module. It is the artefact that lets the ~40 hand-written window
filters the spec implies (§13.10) become ~40 lines of declaration that the
framework hoists into ONE frame-tier pass per timeline.

Four fields carry all the weight:

  grain      the key the window is computed over. contacts are capped PER CLIENT
             (spec §5.5) while treatments are sequenced PER ACCOUNT, so the same
             physical event table is declared twice at two grains. Getting this
             wrong is the 340 000-client harassment bug.

  event_date when the thing happened. Every window's upper bound defaults to
             `decision_date` — NOT to now(). There is no now() reachable from
             any tier of this project (see FRAMEWORK-DEMANDS #1).

  known_at   when the Bank learned of it. This one field is the whole answer to
             spec §13.3. A window filters on
                 event_date <= decision_date  AND  known_at <= knowledge_cutoff
             `knowledge_cutoff` is a shared param. Set it to the run's watermark
             and you get the as-known-on-the-day derivation that audit needs; set
             it to Timestamp.MAX and you get the as-at-now derivation analysis
             wants. ONE implementation, two values of one field. The 2.4% of
             account-days on which they disagree is
                 decider2.impact(as_known, as_at_now, sample)

  late_by_p99 the declared lateness budget. The framework uses it to decide when
             a sequence checkpoint (path/sequence.py) is safe to resume from
             rather than replay. Agency contact outcomes land T+2, so any
             checkpoint younger than 2 days is provisional.
"""

from decider2 import Timeline
from decider2.types import Date, Timestamp

from . import schemas as S

# ---------------------------------------------------------------- contacts ---

CONTACTS_BY_CLIENT = Timeline(
    name="contacts_client",
    schema=S.ContactAttempt,
    grain="client_id",
    event_date="attempted_on",
    known_at="landed_at",
    late_by_p99="2d",              # agency files T+2
    retention="400d",
)

CONTACTS_BY_ACCOUNT = Timeline(
    name="contacts_account",
    schema=S.ContactAttempt,
    grain="account_id",
    event_date="attempted_on",
    known_at="landed_at",
    late_by_p99="2d",
    retention="400d",
)

# ---------------------------------------------------------------- payments ---

PAYMENTS = Timeline(
    name="payments",
    schema=S.PaymentTransaction,
    grain="account_id",
    event_date="value_date",
    known_at="posted_at",
    late_by_p99="36h",             # one of four rails lags 36h
    retention="800d",
)

# ------------------------------------------------- promises / arrangements ---

PROMISES = Timeline(
    name="promises",
    schema=S.Promise,
    grain="account_id",
    event_date="captured_on",
    known_at="captured_on",        # captured in-session; never late
    late_by_p99="0h",
    retention="800d",
)

ARRANGEMENT_INSTALMENTS = Timeline(
    name="arrangement_instalments",
    schema=S.ArrangementInstalment,
    grain="arrangement_id",
    event_date="due_on",
    known_at="settled_at",
    late_by_p99="36h",
    retention="1200d",
)

# ---------------------------------------------------- status and legal feed ---

STATUS_EVENTS = Timeline(
    name="status_events",
    schema=S.StatusEvent,
    grain="client_id",             # debt review is a CLIENT state, not an account state
    event_date="effective_on",
    known_at="received_at",
    late_by_p99="14d",             # the registry short-delivers; see suspensions/status_feeds.py
    retention="4000d",
)

# ------------------------------------------- this flow's own past decisions ---

TREATMENTS = Timeline(
    name="treatments",
    schema=S.TreatmentInstance,
    grain="account_id",
    event_date="decision_date",
    known_at="decision_date",      # our own output; known the moment it is written
    late_by_p99="0h",
    retention="2600d",             # 7 years
)

BUCKET_ROLLS = Timeline(
    name="bucket_rolls",
    schema=S.BucketObservation,
    grain="account_id",
    event_date="observed_on",
    known_at="observed_on",
    late_by_p99="0h",
    retention="800d",
)
