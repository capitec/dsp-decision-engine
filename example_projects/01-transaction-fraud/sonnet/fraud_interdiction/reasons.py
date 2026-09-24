"""Reason ranking and client-facing wording (spec 01 §5.14) -- `core.reason_codes` reused verbatim.

Fraud wording is deliberately less specific than credit wording (telling a
client which velocity threshold they crossed tells the fraudster the same
thing), so the client-safe text is generic per action, not per rule
(§5.14: "less specific... internal reason set and the external wording
used, and the mapping between them"). SCOPE.md skips the multi-language
requirement; the mapping itself -- which is the part §5.14 actually
tests -- is built.
"""
from __future__ import annotations

from decider import missing_as, step

from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry

from fraud_interdiction import vocab

_FAMILY_BASE = {"CF": 1000, "AT": 2000, "MS": 3000, "FP": 4000, "AA": 5000}
_FAMILY_SEVERITY_OFFSET = {"AA": 0, "MS": 10, "AT": 20, "CF": 30, "FP": 40}  # AML-adjacent ranks most severe

REASON_REGISTRY = ReasonCodeRegistry("fraud-reasons-2026.01", [
    ReasonCode(base + i, _FAMILY_SEVERITY_OFFSET[family] + (i % 10), f"{vocab.FAMILY_NAMES[family]} indicator {i}",
               family == "AA")
    for family, base in _FAMILY_BASE.items() for i in range(40)
])

# One client-safe wording per action, not per rule (§5.14's "less specific" requirement).
CLIENT_WORDING = {
    vocab.ACTION_FREEZE_ACCOUNT: "wording.freeze_account.v1",
    vocab.ACTION_BLOCK_CHANNEL: "wording.block_channel.v1",
    vocab.ACTION_DECLINE: "wording.decline.v1",
    vocab.ACTION_HOLD_FOR_REVIEW: "wording.hold_for_review.v1",
    vocab.ACTION_STEP_UP: "wording.step_up.v1",
    vocab.ACTION_MONITOR: "wording.none.v1",
    vocab.ACTION_ALLOW: "wording.none.v1",
}
CLIENT_VISIBLE_ACTIONS = {vocab.ACTION_FREEZE_ACCOUNT, vocab.ACTION_BLOCK_CHANNEL, vocab.ACTION_DECLINE,
                          vocab.ACTION_HOLD_FOR_REVIEW, vocab.ACTION_STEP_UP}
WORDING_REGISTRY_VERSION = "fraud-wording-2026.01"


def client_wording(action_code: int, channel_permitted: bool = missing_as(True)) -> tuple[str, bool]:
    """The wording key shown/spoken to the client, and whether this action is client-visible at all."""
    visible = action_code in CLIENT_VISIBLE_ACTIONS
    if not visible:
        return CLIENT_WORDING[vocab.ACTION_ALLOW], False
    if not channel_permitted:
        return "wording.suppressed_no_consent.v1", True
    return CLIENT_WORDING[action_code], True


client_wording_step = step(client_wording, outputs=("client_wording_key", "client_visible"))
