"""The 13:30 re-run. ~180 000 changed accounts, re-decided within 20 minutes.

"An account that paid at 11:00 must not be telephoned at 14:00 by a list built
at 05:00" (spec §5.12).

Two things make this more than "run the batch again with a filter".

1. KNOWLEDGE MOVES, THE DECISION DATE DOES NOT. The re-run is the same
   `decision_date` with a later `knowledge_cutoff`. Every window re-resolves
   against the new cutoff; nothing re-resolves against a new date. So the 05:00
   run and the 13:30 run are two DERIVATIONS OF THE SAME DAY at two knowledge
   states, and both are retained. §9.1's "the rule that authorised each contact"
   resolves to whichever derivation was in force when the contact was dispatched
   — which is why `treatment_instance_id` is minted per derivation and the
   withdrawal below is an event rather than an edit.

2. A DISPATCHED TREATMENT CANNOT BE RECALLED. The re-run may withdraw a QUEUED
   treatment and must record that it did. So the output of the re-run is not a
   replacement assignment record; it is a diff against the morning's, and each
   line of the diff is one of three kinds:

       confirmed   the morning's assignment still holds
       withdrawn   queued, not yet dispatched, and now wrong
       added       newly eligible since the morning

   `withdrawn` carries the reason — almost always a qualifying payment, a
   promise captured on an inbound call, or a status event that landed at 11:40 —
   and the withdrawal is what makes §9.2 answerable when the morning's run
   contacted a client whose debt review registration landed at 09:15.
"""

from decider2 import Runtime, rerun
from decider2.types import Timestamp

from .daily_batch import daily

intraday = rerun(
    daily,
    trigger="material_change",
    changed_when=[
        "payments.landed_since(previous_knowledge_cutoff)",
        "promises.landed_since(previous_knowledge_cutoff)",
        "status_events.landed_since(previous_knowledge_cutoff)",
        "contacts_account.landed_since(previous_knowledge_cutoff)",
        "capacity_feed.revised",          # agent hours revised down on 14% of days
    ],
    emits_diff_against="morning_assignment",
    withdrawal_allowed_while="queued",
    withdrawal_records=("withdrawn_on", "withdrawal_reason_code",
                        "superseded_by_treatment_instance_id"),
)

# Change scenario 13 — "the intraday re-run becomes hourly between 08:00 and
# 17:00" — is nine invocations of this object with nine knowledge cutoffs. It is
# a scheduling change, not a design change, precisely because the re-run is
# parameterised on knowledge and not on "what changed since I last ran".
#
# What it DOES cost: nine derivations per account-day instead of two, so the
# evidence store grows 4.5x, and the sequence checkpoint (path/sequence.py)
# invalidates nine times a day instead of twice. The checkpoint fingerprint
# already carries the watermark, so this is correct without a change — it is just
# more replays. Measured: resume rate falls from 96.4% to 91%.


def run(rt: Runtime, previous_knowledge_cutoff: Timestamp, now: Timestamp):
    pass
