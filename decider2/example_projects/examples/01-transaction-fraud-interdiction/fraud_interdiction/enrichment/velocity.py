"""5.4 — velocity aggregate attachment.

168 precomputed aggregates arrive from a streaming store. They are not computed
here. What *is* computed here is their **state**, and that is the part the spec
spends a page on:

    "Stale is not missing and missing is not zero. A client who made no payments
     in the last hour and a client whose hourly count could not be retrieved
     must be distinguishable by any rule reading the value, and the distinction
     must survive into the decision record."  (spec §4.3)

Doc 03 §1 offers three null tiers: required, `missing_as(fill)`, and
`float | None`. All three are one bit wide. This project needs four states and a
watermark, so `Observed[T]` replaces them here. See FRAMEWORK-DEMANDS.md #6.

    Observed[T] = NamedTuple(value: T, state: int8, watermark: int64)
    state ∈ {FRESH, STALE, ABSENT, NOT_APPLICABLE}

NOT_APPLICABLE is the fourth and is *not* a data-quality state: it means the
field is structurally absent for this event type (a login has no beneficiary
velocity). Collapsing it into ABSENT would make an aggregate outage and a login
indistinguishable in the outcome statistics, which is exactly the failure
§5.4 is written to prevent.
"""

from __future__ import annotations

from decider2 import Observed, module, observed, param
from decider2.types import Instant

# The 168 aggregates are 7 keys x 6 windows x 4 statistics. They are declared as
# a generated family rather than 168 hand-written functions: the shape is a
# cross-product and writing it out would be 168 near-identical steps, which is
# the "79 identity-passthrough functions" failure mode in miniature.

VELOCITY_KEYS = ("client", "card", "device", "beneficiary", "merchant", "ip", "client_event_type")
VELOCITY_WINDOWS = ("1min", "10min", "1h", "24h", "7d", "30d")
VELOCITY_STATS = ("count", "amount_sum", "distinct_counterparties", "distinct_devices")


def velocity_family() -> dict[str, Observed[float]]:
    """Declare the 168 `Observed` columns as one family.

    Each is `observed(source="velocity", stale_after=<per-window tolerance>)`.
    The tolerance is a *param*, so Fraud Engineering can retune all six windows
    without a compile:

        1min -> 2s,  10min -> 10s,  1h -> 60s,  24h -> 60s,
        7d -> 300s,  30d -> 900s
    """
    pass  # emit 168 declared inputs, named f"{key}_{stat}_{window}"


def velocity_stale_count(**velocity: Observed[float]) -> int:
    """How many of the 168 are stale for their own window's tolerance."""
    pass


def velocity_absent_count(**velocity: Observed[float]) -> int:
    """How many of the 168 could not be retrieved at all."""
    pass


def velocity_short_window_absent(**velocity: Observed[float]) -> bool:
    """Any absent aggregate in the 1-minute or 10-minute windows.

    Called out separately because §5.4 makes it sufficient on its own for
    materially-degraded, regardless of the absent count.
    """
    pass


def velocity_completeness_band(
    velocity_stale_count: int,
    velocity_absent_count: int,
    velocity_short_window_absent: bool,
    materially_degraded_absent_count: int = param(10, ge=1, le=168, unit="count"),
) -> int:
    """0 complete, 1 partially degraded, 2 materially degraded (spec §5.4)."""
    pass


Velocity = module(
    velocity_family,
    velocity_stale_count,
    velocity_absent_count,
    velocity_short_window_absent,
    velocity_completeness_band,
    name="velocity",
    contract="contracts/feature_vector.json#/velocity",
    taps=["velocity_completeness_band"],
)
