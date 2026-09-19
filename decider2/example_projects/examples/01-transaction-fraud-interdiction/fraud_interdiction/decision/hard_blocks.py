"""5.7 — hard blocks.

Six gates that force an action floor. The subtlety the spec insists on:

    "These short-circuit the **action** — no rule and no overlay can soften
     them — but not the **evidence**. The full live and shadow set is still
     evaluated and the complete firing set still recorded."

So a hard block is emphatically NOT an early exit, and it is NOT a `Branch`
around the rule evaluation. Doc 03 §8.2 would make it one — a branch compiles to
a real branch and only the taken arm executes, which is normally the point and
here is precisely wrong. Modelling it as a branch would skip the evaluation the
dispute pack needs.

It is a **floor fed into the resolver**: one more input to the argmax, computed
before evaluation, consumed after it. The rules run regardless.
See FRAMEWORK-DEMANDS.md #14.
"""

from __future__ import annotations

from decider2 import Observed, module, param
from decider2.tables import DecisionTable

GATE_SANCTIONS = 1 << 0
GATE_ACCOUNT_FROZEN = 1 << 1
GATE_CONFIRMED_MULE = 1 << 2
GATE_BLOCKED_DEVICE = 1 << 3
GATE_CARD_COMPROMISED = 1 << 4
GATE_LEGAL_HOLD = 1 << 5

# Six rows, one outcome column, uniform operators — a generic-kernel decision
# table (doc 08 §3.4). Adding a seventh gate is a document change, not a code
# change, and it renders as a six-row grid the Head of Fraud signs.
HARD_BLOCK_TABLE = DecisionTable(
    name="hard_blocks",
    conditions=["gate_bit"],
    outcomes=["forced_action_code", "reason_code", "obligation"],
    document="config/interdict/hard_blocks.json",
)


def hard_block_code(
    sanctions_membership: Observed[int],
    account_state_code: int,
    mule_list_membership: Observed[int],
    device_blocklisted: Observed[bool],
    card_state_code: int,
    legal_hold_active: bool,
) -> int:
    """Bitset of every gate that held. All of them, not the first.

    "Where two gates hold, both are recorded and the more severe action
    applies." A first-match decision table cannot express that; this is a
    bitset and a max, which is the same shape as the rule set itself.
    """
    pass


def hard_block_action_floor(hard_block_code: int) -> int:
    """The most severe forced action across every gate that held.

    Feeds `Resolve` as a floor. Nothing downstream may produce a less severe
    outcome; an overlay cannot touch it, because the overlay gain vector is
    applied to rule tunables and this is not one.
    """
    pass


def hard_block_evidence(hard_block_code: int) -> tuple[int, ...]:
    """Per gate that held: the list version and the matching entry id.

    A sanctions block later found to be a false name match has to be explainable
    in terms of everything else that was true, so this carries the entry, not
    just the fact.
    """
    pass


HardBlocks = module(
    hard_block_code,
    hard_block_action_floor,
    hard_block_evidence,
    name="hard_blocks",
    taps=["hard_block_code"],
)
