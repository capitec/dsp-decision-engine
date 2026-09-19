"""The daily delta — 150 000 to 400 000 changed clients, all live campaigns,
30 minutes, five times a week.

Two things make this a different pipeline rather than the same one with a smaller
frame, and both are requirements rather than optimisations.

1.  **It must use the same feature definitions as the monthly cycle it sits
    between** (spec §5.1 req 3).  A client processed in a delta and again in the
    next monthly run must not be treated under different definitions.  So a delta
    does not resolve its own artefact versions; it **inherits the manifest**:

        cycle = Cycle(..., manifest_inherits="2026-09-M", cycle_date=date(2026,9,17))

    Tree versions, band set, feature registry, suppression registry and overlay
    stack all come from the monthly manifest.  `cycle_date` moves; nothing else
    does.  A tree published on the Tuesday is live in Wednesday's delta only
    because publication writes a manifest amendment, which is a reviewed act
    (see pipelines/publication.py), not because the delta re-resolved "latest".

2.  **Capacity and fatigue are consumed, not reset.**  `Arbitration` runs against
    the month's remaining capacity, carried forward by cycle id.  Scenario 9 —
    six campaigns moving from monthly to daily — is exactly this interaction, and
    it is visible here because the carried input is declared.

Volume: 400 000 x 60 = 24 M evaluations.  Two chunks, not 64.  Pre-assessments
are valid 10 days for daily campaigns against 35 for monthly (spec §5.5 req 3),
so most deltas join an existing pre-assessment rather than requesting one.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from decider2 import Cycle, parallel
from decider2.frame import Filter, Join

from modules.arbitration.allocate import Arbitration
from modules.fatigue import AttachFatigueState
from pipelines.monthly_cycle import campaign_stage

cycle = Cycle(
    name="campaign_targeting_delta",
    cycle_date=...,
    chunks=2,
    chunk_by="stable_hash64(client_id)",
    resume="chunk",
    manifest_inherits="2026-09-M",           # THE line that makes §5.1 req 3 true
)

pipeline = (
    Filter(pl.col("changed_since") >= pl.col("last_cycle_at"))
    | Join("feature_mart", on="client_id", how="inner", snapshot="frozen")
    | AttachFatigueState
    | parallel(*[campaign_stage(c, cycle.cycle_date) for c in cycle.live_campaigns()])
    | Join("project03_preassessment", on=["client_id", "product_code"], how="left",
           missing="request_batch")          # only the 10-day-expired ones
    | Arbitration.with_carried_capacity(consumed_by=cycle.month_to_date())
)

# The shadow evaluation is deliberately ABSENT here.  Spec §8 requires the
# overlays-off twin inside the six-hour monthly window; requiring it inside 30
# minutes as well would double the delta's tree cost for a measurement nobody
# reads daily.  This is a stated omission with an owner, not an oversight — if
# Campaign Analytics wants daily unadjusted volumes the line is one `shadow(...)`
# and the delta budget goes to ~40 minutes.
