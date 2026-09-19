"""Contact fatigue state — the one input a replay cannot reconstruct.

Fatigue is evaluated against contacts *already made*, including those issued by
daily deltas since the last monthly run (spec §5.2).  So the March monthly cycle's
answer depends on the February monthly cycle's output and on the ~20 delta cycles
between them.  Nothing in the feature snapshot carries it.

That makes it a **declared carried input**: an input to the pipeline that is an
output of a previous run of the same pipeline, versioned by the cycle that
produced it and pinned in the cycle manifest.

    CarriedContactState = carried(
        name="contact_state",
        produced_by="monthly_cycle.dispatch",
        key=["client_id"],
        columns=["contacts_30d_all", "contacts_30d_sms", "contacts_30d_call",
                 "contacts_30d_email", "contacts_30d_inapp",
                 "last_contact_date", "last_contact_by_campaign_family"],
        version_column="as_of_cycle_id",
    )

The cycle manifest then records `contact_state@2026090 3`, and a replay of the
March cycle in 2029 resolves that exact state rather than the state as it stands
in 2029.  Without the declaration the replay silently succeeds with today's
fatigue counts and produces a plausible wrong answer — the failure class 00 §7.3
names for dates and which applies identically here.  DEMANDS #30.

Scenario 9 — six campaigns move from monthly to daily — is the reason this must
be a first-class declaration rather than a join the project happens to write:
fatigue and capacity now interact across cycles within a month in a way they did
not before, and the interaction has to be visible in the pipeline, not buried.
"""

from __future__ import annotations

from decider2 import carried, module
from decider2.frame import Join

CarriedContactState = carried(
    name="contact_state",
    produced_by="monthly_cycle.dispatch",
    key=["client_id"],
    version_column="as_of_cycle_id",
)

AttachFatigueState = Join(
    CarriedContactState, on="client_id", how="left",
    missing="zero",              # never contacted is zero contacts, not null
    partition="cycle",
)
