"""Codes shared by the fraud steps. Everything else in a record uses the names in `credit_core.vocab`."""
from __future__ import annotations

from enum import IntEnum

EVENT_TYPE_INSTANT_PAYMENT = 210
CHANNEL_WEB = 4


class Action(IntEnum):
    """What the bank does about an event, most severe last."""

    ALLOW = 10
    MONITOR = 20
    STEP_UP = 30
    HOLD_FOR_REVIEW = 40
    DECLINE = 50
    BLOCK_CHANNEL = 60
    FREEZE_ACCOUNT = 70


class Mode(IntEnum):
    """`degraded_mode_code`: how much of the enrichment the decision could rely on."""

    NORMAL = 0
    REDUCED = 1
    RESTRICTED = 2
    FAIL_CLOSED = 3


class VelocityBand(IntEnum):
    COMPLETE = 0
    PARTIALLY_DEGRADED = 1
    MATERIALLY_DEGRADED = 2


# A velocity aggregate is fresh, stale (older than its tolerance) or absent.
FRESH, STALE, ABSENT = 0, 1, 2

# Rule families, most senior first: the third tie-break between rules that ask for the same action.
FAMILIES = ("AA", "MS", "AT", "CF", "FP")
FAMILY_NAMES = {"CF": "card fraud", "AT": "account takeover", "MS": "mule and scam", "FP": "first-party fraud",
                "AA": "AML-adjacent"}

# `enrichment_degradation_code` has one bit per enrichment source.
BIT_CLIENT_PROFILE, BIT_DEVICE_SESSION, BIT_COUNTERPARTY, BIT_VELOCITY = 1, 2, 4, 8
BIT_MODEL_SCORE, BIT_SANCTIONS, BIT_MULE_WATCHLIST = 16, 32, 64

# `hard_block_code` has one bit per gate: its name, the action it forces and its reason code. The kernel
# in `hard_blocks.hard_block_code` needs the actions as plain comparisons; a test keeps the two in step.
GATES = {
    1: ("sanctions", Action.BLOCK_CHANNEL, 6001),
    2: ("account_frozen", Action.DECLINE, 6002),
    4: ("mule_confirmed", Action.DECLINE, 6003),
    8: ("device_blocked", Action.BLOCK_CHANNEL, 6004),
    16: ("card_compromised", Action.DECLINE, 6005),
    32: ("court_order", Action.HOLD_FOR_REVIEW, 6006),
}
