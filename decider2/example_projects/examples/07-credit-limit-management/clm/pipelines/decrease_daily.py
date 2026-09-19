"""The decrease path (s5.7), daily over 4.1 M accounts in 45 minutes.

Change scenario 10 asks for this to run daily on a bureau distress feed while
the increase path stays monthly. It already does: this is a separate pipeline
object with its own cadence, its own artefact selectors and its own governance,
sharing `Book`, `Scoring` and `Temporal` with the programme.

Decreases are not subject to the s5.2 exclusions and not subject to the consent
requirement, which is why this is a pipeline and not a `Branch` inside the
programme. Two directions with different governance are two pipelines.
"""

from decider2 import Partition, parallel, pipeline
from decider2.frame import Union
from clm.decrease.reduction import ApplyDecrease, Reduction
from clm.decrease.triggers import Triggers
from clm.features.banding import Banding
from clm.features.temporal import Temporal
from clm.scoring.behavioural import Scoring
from clm.sources.book import Book
from clm.vocabulary import vocabulary

decrease = pipeline(
    Book
    | Banding
    | Temporal
    | parallel(Scoring)
    | Triggers
    | Partition(
        on="decrease_required",
        taken=Reduction | ApplyDecrease,
        skipped="passthrough",
        rejoin=Union(schema="schemas/decrease_record.json"),
    )
).with_vocabulary(vocabulary)

# What a branch consultant sees when a client asks why their limit was reduced
# (s9.1) is one row of this pipeline's output:
#
#   trigger_codes            {5}
#   primary_trigger_code     5  "over-limit persistence"
#   d05.observed_value       3          (cycles over limit in the last 4)
#   d05.threshold_applied    3
#   d05.observed_day         2026-08-31
#   adjustments_applied      []          (no cut-off shift was in force)
#   notice_class_code        1  immediate
#   jurisdiction_code        1  home market
#   notice_despatched_day    2026-09-01
#   decrease_target_limit_c  1_842_000   R18 420
#   binding_floor_code       1  money owed
#   money_owed.cap_value_c   1_842_000   R18 420 = balance + authorisations + interest
#
# No analyst, no query, one conversation. Every one of those columns is
# `evidence=` on a panel member, which is why none of them can be dropped by a
# consumer who only wanted the new limit.
