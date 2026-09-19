"""Stage 5.4 (part 1) -- segment assignment and challenger selection.

Segment precedence is stated and stable: thin file beats new-to-bank beats
existing client.  That is a three-row ordered table, so it is a `decision_table`
(doc 08 §3.4 -- tabular, generic kernel, free to change), not a nest of `if`s.

Challenger selection is a deterministic hash of `client_id`, NOT of
`application_id` (a client gets consistent treatment across repeat
applications) and NOT a random draw (a replay must reproduce the selection).
The traffic share has moved 5% -> 10% -> 20% -> 5% inside one year, so it is a
param, and it is recorded on every application.
"""

from __future__ import annotations

from decider2 import decision_table, module, param, step
from decider2.hash import stable_hash_u64

SEGMENT_TABLE = decision_table(
    name="segment_precedence",
    key_columns=["bureau_account_count_ever", "bureau_history_months", "internal_months"],
    result_columns=["segment_code", "scorecard_id"],
    order="first_match",  # precedence is the row order and is part of the artefact
    interior="config/flex_loan/interiors/segment_precedence.json",
)


@step(output="in_challenger_sample")
def in_challenger_sample(
    client_id: int,
    challenger_share: float = param(0.10, ge=0.0, le=1.0, owner="model_risk"),
) -> bool:
    """Deterministic 10% of clients, by a stable hash of `client_id`.

    `stable_hash_u64` is framework-provided and its output is frozen across
    releases -- a hash whose value changes when the interpreter or the library
    changes would silently re-randomise the parallel run and destroy four
    years of comparison.  See FRAMEWORK-DEMANDS #19.
    """
    pass  # stable_hash_u64(client_id) < challenger_share * 2**64


AssignSegment = module(
    SEGMENT_TABLE,
    in_challenger_sample,
    name="segment",
    taps=["segment_code", "scorecard_id", "in_challenger_sample", "challenger_share"],
)
