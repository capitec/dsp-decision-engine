"""Event record schemas. Declared so a Timeline's window predicates type-check
before anything runs, and so `known_at` is never an optional afterthought.

Money is scaled int64 CENTS everywhere, per doc 03 §1.2. Note this contradicts
the spec's own vocabulary table (§4.1 declares `arrears_amount` float64) — see
FRAMEWORK-DEMANDS #21.
"""

from decider2 import EventSchema
from decider2.types import Date, Timestamp, i1, i2, i8, cents


class ContactAttempt(EventSchema):
    contact_id: i8
    account_id: i8
    client_id: i8
    treatment_instance_id: i8 | None      # null for inbound and for agency-initiated
    channel_code: i1                      # 1 SMS 2 email 3 in-app 4 IVM 5 voice 6 field 7 letter
    is_spoken: bool                       # voice + field; the 2-per-day cap keys on this
    outcome_code: i1 | None               # null for 4.1% of dialler records — a real state
    attempted_on: Date
    attempted_at: Timestamp
    landed_at: Timestamp                  # dialler hourly, agency T+2, field T+1
    originator_code: i1                   # 1 bank 2 agency 3 field partner
    script_id: i2 | None


class PaymentTransaction(EventSchema):
    payment_id: i8
    account_id: i8
    amount: cents
    value_date: Date
    posted_at: Timestamp
    rail_code: i1                         # rail 3 lags 36h
    is_reversal: bool


class Promise(EventSchema):
    promise_id: i8
    account_id: i8
    contact_id: i8                        # the contact that captured it
    promised_amount: cents
    promised_on: Date                     # the date money is promised FOR
    captured_on: Date
    part_index: i1                        # 0..2, up to 3 scheduled parts
    outcome_code: i1 | None               # kept/partial/broken/superseded; null until graded
    outcome_graded_on: Date | None


class ArrangementInstalment(EventSchema):
    arrangement_id: i8
    account_id: i8
    instalment_index: i2
    due_on: Date
    due_amount: cents
    settled_amount: cents
    settled_at: Timestamp | None


class StatusEvent(EventSchema):
    status_event_id: i8
    client_id: i8
    account_id: i8 | None                 # disputes are per account; debt review is per client
    status_code: i2                       # maps to suspension_code
    stage_code: i1 | None                 # debt review stage 1..4
    effective_on: Date
    received_at: Timestamp
    source_reference: str                 # DR case number, complaint ref, ombud ref
    source_system_code: i1
    cleared_on: Date | None


class TreatmentInstance(EventSchema):
    """This flow's own output, read back as an input. See README §"History is an input"."""

    treatment_instance_id: i8
    account_id: i8
    client_id: i8
    decision_date: Date
    episode_id: i8
    treatment_code: i1
    treatment_intensity: i1
    channel_code: i1 | None
    path_position_before: i1
    path_position_after: i1
    was_dispatched: bool                  # allocation may have cut it below the line
    withdrawn_on: Date | None             # intraday re-run withdrew a queued treatment
    matrix_version: i2
    matrix_cell_id: i2
    cohort_code: i2
    adjustment_set_version: i2


class BucketObservation(EventSchema):
    account_id: i8
    observed_on: Date
    arrears_bucket_code: i1
    days_past_due: i2
    banding_version: i2                   # cell ids are only comparable within a banding version
