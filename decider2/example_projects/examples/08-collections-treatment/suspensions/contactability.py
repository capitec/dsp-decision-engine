"""Suspensions 115-120 — the ones that are about the CLIENT, not the account.

115 permitted contact hours     117 frequency cap
116 channel consent withdrawn   118 written communication only
119 promise in force            120 no valid contact point

Two of these (117, 118) and the caps behind 115 are PER CLIENT. Spec §5.5:
"Caps are per client, not per account, which matters for the 340 000 clients with
more than one delinquent account — three accounts each politely observing a
per-account cap produce a harassment complaint."

This is the sharpest grain bug available in this project and it is invisible at
review time, because every rule below reads perfectly well as an account rule.
The defence is mechanical: the windows these read are declared on
CONTACTS_BY_CLIENT (timelines/streams.py) and entered the state vector through
`.broadcast(from_grain="client_id")` (state/assembly.py). A rule reading a
client-grain value that was not explicitly broadcast is a build error.

But a broadcast value read by three sibling accounts is still only HALF the
answer, because all three will independently conclude "one more contact is
permitted today". The other half is a per-client QUOTA in the allocator
(allocation/constraints.py), which is why non-selection code 260 — "coverage
constraint already satisfied by another account of the same client" — exists at
all. Per-record logic cannot decide a per-client cap; it can only decide
eligibility for one.
"""

from decider2 import param, suspension
from decider2.types import Date, Timestamp, i1, i2, i4

from ..timelines import windows as W
from ..timelines.calendar import add_bdays
from .panel import Scope, boundary, computed, event

# Compliance-owned tables. Permitted contact hours is 7 days x 4 channel groups
# x 2 edges; frequency caps are 4 channel groups x 5 windows x {per-client,
# per-account}. Both on Compliance's release path, both effective-dated, and
# NEITHER is an OverlaySurface — see overlays/guard.py.
from ..matrix.grid import CONTACT_HOURS, FREQUENCY_CAPS


@suspension(
    code=115,
    name="contact_hours",
    description="Outside permitted contact hours or days for this channel group.",
    scope=Scope.blocks_channels(5, 6),     # spoken channels; asynchronous channels queue
    expiry=boundary("next_permitted_window"),
    owner="regulatory_compliance",
)
def contact_hours(
    decision_date: Date,
    channel_group_code: i1,
    client_preferred_window_code: i1 | None,
    day_of_week: i1,
) -> bool:
    """Evaluated against the WINDOW, not against a clock.

    The batch runs at 04:20 and decides what may happen at 09:00, 13:00 and
    18:30. So this rule does not answer "is it permitted now" — there is no now.
    It answers "is there any permitted window today", and the output assembly
    (evidence/record.py) carries the earliest and latest permitted times as the
    dispatch instruction. The dialler enforces the instant; the flow decides the
    envelope, and only the envelope is replayable.
    """
    pass


@suspension(
    code=116,
    name="consent_withdrawn",
    description="The client has withdrawn consent for this channel.",
    scope=Scope.blocks_channels("per_consent_state"),
    expiry=event(awaits="consent.re_obtained", reviewed="daily", feed="consent"),
    owner="regulatory_compliance",
)
def consent_withdrawn(
    sms_consent_withdrawn_at: Timestamp | None,
    email_consent_withdrawn_at: Timestamp | None,
    voice_consent_withdrawn_at: Timestamp | None,
) -> bool:
    pass  # any channel withdrawn; the SCOPE narrows per channel, the bit is set once


@suspension(
    code=117,
    name="frequency_cap",
    description="A contact frequency cap has been reached for this client.",
    scope=Scope.blocks_channels(1, 2, 3, 4, 5, 6),
    expiry=computed("frequency_cap_lifts_on"),
    owner="regulatory_compliance",
)
def frequency_cap(
    contacts_today_client: i2,
    spoken_today_client: i2,
    contacts_7d_client: i2,
    contacts_30d_client: i2,
    cap_per_day: i2 = param(3, ge=1, le=10, owner="regulatory_compliance"),
    cap_spoken_per_day: i2 = param(2, ge=1, le=6, owner="regulatory_compliance"),
    cap_per_7d: i2 = param(10, ge=1, le=40, owner="regulatory_compliance"),
    cap_per_30d: i2 = param(24, ge=1, le=120, owner="regulatory_compliance"),
) -> bool:
    """Change scenario 3 — Compliance tightens 10-per-7-days to 6, per client
    rather than per account, effective on a date, with the old rule governing
    everything before it — is FOUR changes to this one declaration:
      cap_per_7d 10 -> 6            (a value, effective-dated)
      grain already per-client      (nothing to do; it was never per-account)
      an effective-from date        (the params document is effective-dated)
      old rule governs before it    (replay resolves params by decision_date)
    None of them is a code change, and the second is the reason to have got the
    grain right on day one rather than at the ombud's request.
    """
    pass


def frequency_cap_lifts_on(
    contacts_7d_client: i2, contacts_30d_client: i2, oldest_contact_in_7d_on: Date | None,
) -> Date | None:
    """The cap lifts when the oldest contact falls out of the rolling window.
    Computable, which is what §5.2 demands of every expiry."""
    pass


@suspension(
    code=118,
    name="written_only",
    description="Client has requested written communication only.",
    scope=Scope.blocks_channels(4, 5, 6),
    expiry=event(awaits="client.withdraws_written_only", reviewed="daily", feed="consent"),
    owner="regulatory_compliance",
)
def written_only(written_only_requested_at: Timestamp | None,
                 written_only_withdrawn_at: Timestamp | None) -> bool:
    pass


@suspension(
    code=119,
    name="promise_in_force",
    description="A promise to pay is in force. Everything except the permitted "
                "promise reminder is held until the promise date plus grace.",
    scope=Scope.blocks_all_except(1, 3),   # the permitted reminder only
    expiry=computed("promise_grace_ends_on"),
    owner="collections_strategy",
)
def promise_in_force(open_promise_date: Date | None, promise_grace_ends_on: Date | None,
                     decision_date: Date) -> bool:
    pass


def promise_grace_ends_on(
    open_promise_date: Date | None,
    bday,
    grace_business_days: i2 = param(2, ge=0, le=10, owner="collections_strategy"),
) -> Date | None:
    """promise date + 2 BUSINESS days. A promise for Friday is graded Tuesday,
    not Sunday. One call, no branch, because the calendar is a dense index."""
    pass  # add_bdays(open_promise_date, grace_business_days, bday)


@suspension(
    code=120,
    name="no_contact_point",
    description="No valid contact point exists on any permitted channel.",
    scope=Scope.blocks_channels(1, 2, 3, 4, 5),
    expiry=event(awaits="contact_point.validated", reviewed="daily", feed="consent"),
    owner="collections_operations",
)
def no_contact_point(
    has_valid_mobile: bool,
    has_valid_email: bool,
    is_app_active: bool,
    wrong_number_by_channel_90d: i2,
    wrong_numbers_to_invalidate: i2 = param(3, ge=1, le=10, owner="collections_strategy"),
) -> bool:
    """Three wrong-number outcomes invalidate the contact point rather than
    escalating the account (spec §5.1). This is the only place in the project
    where a contact OUTCOME feeds contactability rather than sequence, and
    keeping it here rather than in path/ is what stops "no answer" and "wrong
    number" being folded together — they reach different modules."""
    pass
