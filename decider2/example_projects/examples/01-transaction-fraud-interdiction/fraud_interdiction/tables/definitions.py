"""The sixteen tables, declared in one place.

Five owners, cadences from quarterly to every fifteen minutes, sizes from 9 rows
to 3.2 M, and two access patterns that are not the same kind of thing at all:

  * `versioned_table` — `key -> row`, resolved against a version effective at
    `event_timestamp`. Cell-level attribution, diffable, editable by an analyst.
    Sixteen of these would be doc 03's `Table` sketch, and it is adequate.

  * `temporal_table` — `(key, instant) -> row`, where the question is
    *membership at a microsecond in the past*, the list replaces itself 24 times
    a day, and the answer must be identical in 2031.

The second is the one doc 03 has no notion of, and it is the difference between
a 2 M-row table and 61 320 copies of one. See FRAMEWORK-DEMANDS.md #17.

Storage follows from the declaration, not from the author: `backing` is chosen
from declared row count and cadence, and the same `MCC_RISK[mcc].risk_band`
expression reads a 1 024-row dense array and a 3.2 M-row hash index without the
step knowing which. What the author declares is the *contract* — cadence,
owner, replay requirement — and those are governance facts, not storage facts.
"""

from __future__ import annotations

from decider2.tables import IntervalStore, temporal_table, versioned_table

# --- small, analyst-edited, diffable ---------------------------------------

MCC_RISK = versioned_table(
    name="merchant_category_risk", key="mcc", rows=1024, attributes=6,
    owner="fraud-strategy", cadence="weekly",
    document="tables/data/merchant_category_risk.csv",
    replay="version_per_event", diffable=True,
)

COUNTRY_RISK = versioned_table(
    name="country_risk", key="country_code", rows=249, attributes=5,
    owner="financial-crime-compliance", cadence="monthly",
    emergency_cadence="same_day",      # spec §11.16
    document="tables/data/country_risk.csv",
    replay="version_per_event", diffable=True,
)

DEVICE_BANDS = versioned_table(
    name="device_reputation_bands", key="band", rows=12, attributes=7,
    owner="fraud-strategy", cadence="quarterly",
    replay="version_per_event", diffable=True,
)

BENEFICIARY_AGE_BUCKETS = versioned_table(
    name="beneficiary_age_buckets", key="bucket", rows=9, attributes=2,
    owner="fraud-strategy", cadence="rarely",
    replay="version_per_event", diffable=True,
)

SEGMENTS = versioned_table(
    name="segment_definitions", key="segment_id", rows=46, attributes=4,
    owner="fraud-strategy", cadence="monthly",
    replay="version_per_event", diffable=True,
    # Spec §11.15: the youth upper age moves 24 -> 26 and 1.9 M clients change
    # segment overnight. Events before the change replay against the old
    # definition, which works only because segment *membership* is computed from
    # the versioned definition at event_timestamp and recorded as a bitmask —
    # not read from a mutable membership table.
)

# --- large, hot, and not analyst-edited ------------------------------------

BIN_ISSUER = versioned_table(
    name="bin_issuer", key="bin_prefix", rows=50_000, attributes=9,
    owner="cards-operations", cadence="monthly", source="scheme_file",
    replay="version_per_event", diffable=True, backing="dense_sorted",
)

MERCHANT_REPUTATION = versioned_table(
    name="merchant_reputation", key="merchant_id", rows=3_200_000, attributes=4,
    owner="fraud-strategy", cadence="daily", source="internal+consortium",
    replay="version_per_event", diffable="summary_only", backing="hash_index",
    # A 3.2 M-row daily diff is not reviewable as a change list, so `diffable`
    # degrades honestly to a summary (counts by band transition) rather than
    # pretending. Declaring that is better than discovering it.
)

# --- membership at an instant ----------------------------------------------
# Stored as intervals, never as snapshots. An hourly refresh that changes 0.3%
# of a 2 M-row list appends ~6 000 rows. Seven years is ~370 M interval rows,
# which is a partitioned parquet dataset, not 61 320 table copies.

TEMPORAL = IntervalStore(
    name="fraud_membership",
    retention_days=2555,
    tables=[
        temporal_table(
            name="mule_watchlist", key="identifier", attributes=5,
            owner="financial-crime-compliance", refresh="hourly",
            sources=("internal", "consortium", "law_enforcement"),
            on_unavailable="fail_closed_on_last_good",
            last_good_tolerance_s=7200,
            # spec §11.9: 2 M -> 11 M entries, hourly -> every 5 minutes. The
            # interval representation scales with the CHANGE rate, so a 5.5x
            # bigger list refreshed 12x more often is ~2x the append volume,
            # not 66x.
        ),
        temporal_table(
            name="sanctions", key="party_id", attributes=6,
            owner="financial-crime-compliance", refresh="15min",
            on_unavailable="fail_closed",
        ),
        temporal_table(
            name="blocked_devices", key="device_id", attributes=3,
            owner="fraud-strategy", refresh="continuous",
            on_unavailable="fail_closed_on_last_good",
        ),
        temporal_table(
            name="compromised_cards", key="card_token", attributes=4,
            owner="cards-operations", refresh="continuous",
            on_unavailable="fail_closed_on_last_good",
        ),
        temporal_table(
            name="legal_holds", key="client_id", attributes=5,
            owner="legal", refresh="continuous",
            on_unavailable="fail_closed",
            # spec §11.17: a court order holding one client for 90 days, then
            # reverting automatically. An interval with an end instant does
            # that without anyone remembering to remove it.
        ),
    ],
)


def assert_replayable(store: IntervalStore, at_instant, key) -> None:
    """The statement a dispute pack needs, produced from the interval:

        'on the list at 14:22:03.417 on 3 March 2026, added 09:11 on 2 March
         2026 by the consortium feed, removed 11:00 on 18 April 2026'

    Note what is NOT permitted: answering from the current list. A replay that
    reaches for a current value has failed (spec §9.2), and that is enforceable
    here because `member_at` has no signature that omits the instant.
    """
    pass
