"""Local vocabulary (spec 01 §4.4) -- declared here because `core` does not publish it.

From `core`, verbatim (imported by callers, not redeclared): `client_id`,
`channel_code`, `product_code`, `decision_date`, `decline_reason_codes`,
`primary_reason_code`.
"""
from __future__ import annotations

from enum import IntEnum

# --- Event types in scope for this slice (§4.1) -- instant payments only ---
EVENT_TYPE_INSTANT_PAYMENT = 210

# --- Action precedence (§5.12), most severe first ---
ACTION_FREEZE_ACCOUNT = 70
ACTION_BLOCK_CHANNEL = 60
ACTION_DECLINE = 50
ACTION_HOLD_FOR_REVIEW = 40
ACTION_STEP_UP = 30
ACTION_MONITOR = 20
ACTION_ALLOW = 10

ACTION_NAMES = {
    ACTION_FREEZE_ACCOUNT: "freeze_account",
    ACTION_BLOCK_CHANNEL: "block_channel",
    ACTION_DECLINE: "decline",
    ACTION_HOLD_FOR_REVIEW: "hold_for_review",
    ACTION_STEP_UP: "step_up",
    ACTION_MONITOR: "monitor",
    ACTION_ALLOW: "allow",
}
ACTION_PRECEDENCE_TABLE_VERSION = "action-precedence-2026.01"

# --- Rule status (§6.1) ---
STATUS_LIVE = "live"
STATUS_SHADOW = "shadow"
STATUS_RETIRED = "retired"

# --- Rule families (§6.1) ---
FAMILY_CARD_FRAUD = "CF"
FAMILY_ACCOUNT_TAKEOVER = "AT"
FAMILY_MULE_SCAM = "MS"
FAMILY_FIRST_PARTY_FRAUD = "FP"
FAMILY_AML_ADJACENT = "AA"
FAMILY_NAMES = {
    FAMILY_CARD_FRAUD: "card fraud",
    FAMILY_ACCOUNT_TAKEOVER: "account takeover",
    FAMILY_MULE_SCAM: "mule and scam",
    FAMILY_FIRST_PARTY_FRAUD: "first-party fraud",
    FAMILY_AML_ADJACENT: "AML-adjacent",
}
# Family precedence for the fourth tie-break rung (§5.12 item 5), most senior first.
FAMILY_PRECEDENCE = (FAMILY_AML_ADJACENT, FAMILY_MULE_SCAM, FAMILY_ACCOUNT_TAKEOVER,
                     FAMILY_CARD_FRAUD, FAMILY_FIRST_PARTY_FRAUD)

# --- Stale/absent behaviour a rule declares per referenced feature (§5.4) ---
STALE_ABSENT_EVALUATE_FALSE = "evaluate_false"
STALE_ABSENT_LAST_KNOWN = "last_known_good"
STALE_ABSENT_SUPPRESS = "suppress"  # -> unevaluable

# --- Degraded mode (§5.6) ---
DEGRADED_MODE_NORMAL = "normal"
DEGRADED_MODE_REDUCED = "reduced"
DEGRADED_MODE_RESTRICTED = "restricted"
DEGRADED_MODE_FAIL_CLOSED = "fail_closed"

# enrichment_degradation_code bit positions (§4.4, §5.6) -- one bit per source
BIT_CLIENT_PROFILE = 1 << 0
BIT_DEVICE_SESSION = 1 << 1
BIT_COUNTERPARTY = 1 << 2
BIT_MERCHANT = 1 << 3
BIT_VELOCITY = 1 << 4
BIT_MODEL_SCORE = 1 << 5
BIT_SANCTIONS = 1 << 6
BIT_MULE_WATCHLIST = 1 << 7

# --- Hard block gates (§5.7), reason family and forced action ---
GATE_SANCTIONS = "sanctions"
GATE_ACCOUNT_FROZEN = "account_frozen"
GATE_MULE_CONFIRMED = "mule_confirmed"
GATE_DEVICE_BLOCKED = "device_blocked"
GATE_CARD_COMPROMISED = "card_compromised"
GATE_COURT_ORDER = "court_order"

GATE_FORCED_ACTION = {
    GATE_SANCTIONS: ACTION_BLOCK_CHANNEL,
    GATE_ACCOUNT_FROZEN: ACTION_DECLINE,
    GATE_MULE_CONFIRMED: ACTION_DECLINE,
    GATE_DEVICE_BLOCKED: ACTION_BLOCK_CHANNEL,
    GATE_CARD_COMPROMISED: ACTION_DECLINE,
    GATE_COURT_ORDER: ACTION_HOLD_FOR_REVIEW,
}
# Gate severity for "more severe action applies" when two gates hold (§5.7).
GATE_SEVERITY = {g: a for g, a in GATE_FORCED_ACTION.items()}


class ConfirmationOfPayeeBand(IntEnum):
    NOT_AVAILABLE = 0
    NO_MATCH = 1
    CLOSE = 2
    EXACT = 3
