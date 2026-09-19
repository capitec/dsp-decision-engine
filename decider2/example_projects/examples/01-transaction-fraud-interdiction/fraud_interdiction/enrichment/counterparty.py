"""5.3 — counterparty and beneficiary enrichment.

Where the mule watchlist is read, and therefore where the hardest table
requirement in the spec lands:

    "Was account X on the list at 14:22:03 on 3 March 2026?" must be answerable
    cheaply and exactly, and the answer must be the same in 2031 — for a 2 M-row
    list that replaces itself 24 times a day, retained 7 years.

Storing 24 snapshots a day for seven years is 61 320 copies of a 2 M-row table.
So the list is not stored as snapshots at all. It is stored as **intervals** —
one row per (identifier, added_at, removed_at, source) — and membership at an
instant is an interval containment test. An hourly refresh that changes 0.3% of
a 2 M-row list writes 6 000 rows, not 2 000 000. Spec §11.9's 5x growth and
5-minute refresh multiplies the *change* rate, not the storage.

`temporal_table` is the framework kind that does this. Doc 03's `Table` sketch
is `key -> row`; this is `(key, instant) -> row`. See FRAMEWORK-DEMANDS.md #17.
"""

from __future__ import annotations

from decider2 import Observed, module, observed, param
from decider2.types import Instant

from fraud_interdiction.tables.definitions import TEMPORAL


def beneficiary_age_hours(
    beneficiary_first_seen_at: Observed[Instant],
    event_timestamp: Instant,
) -> Observed[float]:
    """Hours since this beneficiary was added to this client's list."""
    pass


def beneficiary_added_within_60min(beneficiary_age_hours: Observed[float]) -> Observed[bool]:
    """The highest-yield scam interdiction signal in the spec."""
    pass


def beneficiary_first_payment(
    prior_payment_count_to_beneficiary: Observed[int],
) -> Observed[bool]:
    """First payment to this beneficiary. Absent count propagates to absent flag —
    it does not become True, which would fire the scam family on an outage."""
    pass


def mule_list_membership(
    beneficiary_id: str,
    event_timestamp: Instant,
) -> Observed[int]:
    """Membership **as at `event_timestamp`**, with listing source and added-at.

    The recorded form is a statement, not a boolean: "on the list at 14:22 on
    3 March, added 09:11 on 2 March by the consortium feed, removed 11:00 on
    18 April" (spec §5.3). So the tap carries the interval id, and the interval
    store is append-only.

    Fail-closed on the last good copy: if the store is unreachable and the last
    good copy is older than tolerance, this is ABSENT and 5.6 escalates the
    event to fail-closed mode. It never quietly returns "not a member".
    """
    pass  # TEMPORAL.mule_watchlist.member_at(beneficiary_id, event_timestamp)


def sanctions_membership(
    beneficiary_id: str,
    event_timestamp: Instant,
) -> Observed[int]:
    """Sanctions / blocked-party membership as at the instant. Fail closed."""
    pass  # TEMPORAL.sanctions.member_at(...)


def cop_name_match_band(
    beneficiary_id: str,
    beneficiary_name_supplied: str,
) -> Observed[int]:
    """Confirmation-of-payee band: exact / close / no-match / not-available.

    Four values, of which *not-available* is a legitimate band rather than a
    null — the scheme sometimes genuinely cannot answer. This is the case that
    makes a three-state `Observed` insufficient on its own: the domain has its
    own "unknown", distinct from the framework's ABSENT, and conflating them
    would let a CoP outage look like 100% no-match.
    """
    pass


def beneficiary_bank_mule_band(
    beneficiary_bank_code: int,
    event_timestamp: Instant,
) -> Observed[int]:
    """Receiving bank's mule-rate band, 1–5, from the versioned bank table."""
    pass


Counterparty = module(
    beneficiary_age_hours,
    beneficiary_added_within_60min,
    beneficiary_first_payment,
    mule_list_membership,
    sanctions_membership,
    cop_name_match_band,
    beneficiary_bank_mule_band,
    name="counterparty",
    contract="contracts/feature_vector.json#/counterparty",
    taps=["mule_list_membership", "sanctions_membership"],
)
