"""Hard blocks (spec 01 §5.7): gates that set an action floor no rule or overlay can soften.

Membership in the sanctions list and the mule watchlist is stubbed as
already-resolved booleans on the request (SCOPE.md: "stub enrichment"),
each carrying its own list version, exactly as a real membership-as-at
lookup would return -- the "was X on the list at this instant" evidence
requirement (§6.4) is met by recording the version and source alongside
the boolean, not by rebuilding a real 2M-row watchlist for this slice.

Rule evaluation still runs regardless of a hard block (§5.7: "rule
evaluation continued regardless") -- this module only computes the forced
action floor; `action_resolution.py` combines it with the firing set.
"""
from __future__ import annotations

from decider import missing_as, step

from fraud_interdiction import vocab

_GATE_ORDER = (
    vocab.GATE_SANCTIONS, vocab.GATE_ACCOUNT_FROZEN, vocab.GATE_MULE_CONFIRMED,
    vocab.GATE_DEVICE_BLOCKED, vocab.GATE_CARD_COMPROMISED, vocab.GATE_COURT_ORDER,
)


def hard_block_code(
    sanctions_list_match: bool = missing_as(False), sanctions_list_version: str = missing_as(""),
    account_frozen: bool = missing_as(False),
    mule_watchlist_confirmed: bool = missing_as(False), mule_watchlist_version: str = missing_as(""),
    device_blocked: bool = missing_as(False),
    card_compromised: bool = missing_as(False),
    court_order_hold: bool = missing_as(False),
) -> tuple[str, int]:
    """Every gate that held (comma-joined, empty string if none) and the forced action (0 if none)."""
    held = []
    if sanctions_list_match:
        held.append(vocab.GATE_SANCTIONS)
    if account_frozen:
        held.append(vocab.GATE_ACCOUNT_FROZEN)
    if mule_watchlist_confirmed:
        held.append(vocab.GATE_MULE_CONFIRMED)
    if device_blocked:
        held.append(vocab.GATE_DEVICE_BLOCKED)
    if card_compromised:
        held.append(vocab.GATE_CARD_COMPROMISED)
    if court_order_hold:
        held.append(vocab.GATE_COURT_ORDER)
    if not held:
        return "", 0
    forced_action = max(vocab.GATE_FORCED_ACTION[g] for g in held)
    return ",".join(g for g in _GATE_ORDER if g in held), forced_action


hard_block_code_step = step(hard_block_code, outputs=("hard_block_code", "hard_block_forced_action"))
