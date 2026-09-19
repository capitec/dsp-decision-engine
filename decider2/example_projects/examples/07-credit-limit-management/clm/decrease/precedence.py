"""The simultaneous case (s5.7). 480 000 clients hold both products and about
2 900 per cycle qualify for an increase on one account and a decrease on the
other. That is not a contradiction to suppress; it is two accounts behaving
differently, with an explicit precedence.

The precedence is CLIENT level and the decision is ACCOUNT level, so this is the
second place in the project where the frame/record boundary falls, and it falls
for exactly the same reason as the first: a per-record rule cannot see a sibling
row.
"""

import polars as pl
from decider2 import module
from decider2.frame import Aggregate, Join

ClientDecreaseState = Aggregate(
    source="decrease_proposals",
    by="client_id",
    metrics={
        "client_immediate_decrease": (pl.col("notice_class_code") == 1).any(),
        "client_notice_decrease_non_dormant":
            ((pl.col("notice_class_code") == 2) & (pl.col("trigger_codes") != {12})).any(),
        "suppressing_account_id": pl.col("account_id").sort_by("severity_rank").first(),
        "suppressing_trigger_code": pl.col("primary_trigger_code")
                                      .sort_by("severity_rank").first(),
    },
    declares={"client_immediate_decrease": pl.Boolean,
              "client_notice_decrease_non_dormant": pl.Boolean,
              "suppressing_account_id": pl.Int64,
              "suppressing_trigger_code": pl.Int8},
)

AttachClientState = Join(source=ClientDecreaseState, on="client_id", how="left")


def increase_suppressed_by_client(client_immediate_decrease: bool,
                                  client_notice_decrease_non_dormant: bool) -> bool:
    """An immediate-class decrease on any account of the client suppresses every
    increase for that client in the cycle -- deterioration is a client-level
    signal. A notice-class decrease does the same, EXCEPT where the sole trigger
    is D12 dormancy, which is not a risk signal."""
    return client_immediate_decrease or client_notice_decrease_non_dormant


Precedence = module(
    increase_suppressed_by_client,
    name="client_precedence",
    evidence=["increase_suppressed_by_client", "suppressing_account_id",
              "suppressing_trigger_code"],
)
# "You were not offered an increase on your card because your facility went two
# cycles past due" is reconstructable from those three columns and nothing else.
