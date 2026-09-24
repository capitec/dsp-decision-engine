"""Locally declared vocabulary (spec 04 §4.4). `credit_core` does not publish any of this --
it belongs to campaign targeting alone.
"""
from __future__ import annotations

# leaf_outcome_code
DO_NOT_TARGET = 0
TARGET = 1
TARGET_AS_CONTROL = 2  # would-target leaf reached by a control-group client (§5.7 item 4)

# offer_tier_code (one of seven; only a few used by the demo campaign)
TIER_A, TIER_B, TIER_C, TIER_D, TIER_E, TIER_F, TIER_G = range(1, 8)

# channel_code (§6, channel capacity table)
CHANNEL_IN_APP = 1
CHANNEL_SMS = 2
CHANNEL_CALL = 3
CHANNEL_EMAIL = 4
CHANNEL_NAME = {CHANNEL_IN_APP: "in_app", CHANNEL_SMS: "sms", CHANNEL_CALL: "call", CHANNEL_EMAIL: "email"}

# suppression class (§5.2 requirement 3)
SUPPRESSION_ABSOLUTE = "absolute"
SUPPRESSION_MEASUREMENT_RELEVANT = "measurement_relevant"

# arbitration non-selection reasons (§5.6 requirement 3)
LOST_ON_RANK = "lost_on_rank"
CHANNEL_CAPACITY_EXHAUSTED = "channel_capacity_exhausted"
FATIGUE_CAP_REACHED = "fatigue_cap_reached"
IS_CONTROL_GROUP = "control_group"
CAMPAIGN_SUSPENDED = "campaign_suspended"

# reason_label registry (subset of ~220; §5.3.2) -- (label_id, text)
REASON_LABELS = {
    9101: "Prime top-up, digitally engaged",
    9102: "Prime top-up, assisted channel",
    9103: "Standard top-up, SMS responsive",
    9104: "Standard top-up, low channel response",
    9201: "No qualifying facility",
    9202: "Risk below campaign floor",
    9203: "Insufficient instalment headroom",
    9204: "Pre-assessed amount below campaign minimum",
}
