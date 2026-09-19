"""5.6 — the enrichment completeness verdict, and the degraded-mode population
adjustment.

"'It degraded gracefully' is not an acceptable answer to 'what did it do'."
(spec §8)

Everything here is declared data, read from `config/interdict/degraded_modes.json`,
because the suspension set (~40 rules) and the activation set (~15) are owned by
the Head of Fraud on a quarterly cadence and must be changeable without a
release. They are *values*: two bitmasks over the rule index space, selected by
the composite mode code. Changing which rules a mode suspends is a params swap.
"""

from __future__ import annotations

from decider2 import Observed, module, param
from decider2.tables import DecisionTable

MODE_NORMAL = 0
MODE_REDUCED = 1
MODE_RESTRICTED = 2
MODE_FAIL_CLOSED = 3

SOURCE_BITS = {
    "client_profile": 1 << 0,
    "device_session": 1 << 1,
    "counterparty": 1 << 2,
    "merchant": 1 << 3,
    "velocity": 1 << 4,
    "model_card": 1 << 5,
    "model_scam": 1 << 6,
    "sanctions": 1 << 7,
    "mule_watchlist": 1 << 8,
}


def enrichment_degradation_code(
    client_profile_state: int,
    device_state: int,
    counterparty_state: int,
    merchant_state: int,
    velocity_completeness_band: int,
    model_presence_code: int,
    sanctions_state: int,
    mule_list_state: int,
) -> int:
    """Bitset over the nine sources. One int16, recorded on every event."""
    pass


# The mode table is a `decision_table`, not an `if` chain: doc 08 §3.4's generic
# kernel kind. Nine condition columns, four outcome rows, uniform operators — so
# a change to it is free (no recompile) and it renders as a grid a non-engineer
# reads. An `if` chain would have made the same logic a skeleton change.
DEGRADED_MODE_TABLE = DecisionTable(
    name="degraded_mode",
    conditions=["enrichment_degradation_code", "velocity_completeness_band",
                "model_presence_code", "sanctions_last_good_age_s",
                "mule_last_good_age_s"],
    outcome="degraded_mode_code",
    document="config/interdict/degraded_modes.json",
)


def degraded_mode_code(
    enrichment_degradation_code: int,
    velocity_completeness_band: int,
    model_presence_code: int,
    sanctions_last_good_age_s: float,
    mule_last_good_age_s: float,
) -> int:
    """normal / reduced / restricted / fail-closed (spec §5.6)."""
    pass  # DEGRADED_MODE_TABLE scan


def mode_suspended_rule_bits(degraded_mode_code: int) -> tuple[int, ...]:
    """Bitmask over the rule index space: rules suspended by this mode.

    Recorded on the event. "When the post-mortem asks why losses spiked on the
    afternoon the streaming store fell over, the answer must be in that
    afternoon's decision records, not inferred."
    """
    pass


def mode_activated_rule_bits(degraded_mode_code: int) -> tuple[int, ...]:
    """The compensating set — rules that reference no model score and no
    velocity, activated only in restricted and fail-closed mode."""
    pass


Completeness = module(
    enrichment_degradation_code,
    degraded_mode_code,
    mode_suspended_rule_bits,
    mode_activated_rule_bits,
    name="completeness",
    taps=["enrichment_degradation_code", "degraded_mode_code"],
    contract="contracts/feature_vector.json#/degradation",
)
