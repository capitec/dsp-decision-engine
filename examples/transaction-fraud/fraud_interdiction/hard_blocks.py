"""Hard blocks: gates that set a minimum action no rule or overlay can soften."""
from __future__ import annotations

from decider import missing_as, step

from fraud_interdiction.vocab import Action


def hard_block_code(
    sanctions_list_match: bool = missing_as(False), account_frozen: bool = missing_as(False),
    mule_watchlist_confirmed: bool = missing_as(False), device_blocked: bool = missing_as(False),
    card_compromised: bool = missing_as(False), court_order_hold: bool = missing_as(False),
) -> tuple[int, int]:
    """The gates that held, one bit each (`vocab.GATES`), and the most severe action they force (0 for none)."""
    code, forced = 0, 0
    if sanctions_list_match:
        code |= 1
        forced = max(forced, Action.BLOCK_CHANNEL)
    if account_frozen:
        code |= 2
        forced = max(forced, Action.DECLINE)
    if mule_watchlist_confirmed:
        code |= 4
        forced = max(forced, Action.DECLINE)
    if device_blocked:
        code |= 8
        forced = max(forced, Action.BLOCK_CHANNEL)
    if card_compromised:
        code |= 16
        forced = max(forced, Action.DECLINE)
    if court_order_hold:
        code |= 32
        forced = max(forced, Action.HOLD_FOR_REVIEW)
    return code, forced


hard_block_step = step(hard_block_code, outputs=("hard_block_code", "hard_block_forced_action"))
