"""EP-9 -- the RM pre-assessment. 3 000/day, p95 under 2 seconds, no external calls.

    "the entry point most likely to be got wrong, because it looks like a cheap
     version of EP-1 and is not."

Spec 5.1's five rules for EP-9, and the two that are structural rather than
behavioural:

  3. It is NOT a decision of record, and must not enter a facility's decision
     history -- but it IS recorded as an interaction, because the question that
     gets asked three months later is "you told me R3 000 000 in March and
     offered R1 800 000 in June".

  "must be STRUCTURALLY INCAPABLE of writing one" (spec 5.3 O17)

Structurally incapable is a strong claim and it needs a mechanism, not a flag.
`writes_decision_of_record=False` on the entry declaration removes the decision
record writer from the graph entirely: the pipeline does not contain the node,
so there is no code path to it, and `pipeline.lineage("decision_of_record_id")`
returns empty. A flag would be a runtime check that a future refactor deletes.
"""

from decider2 import fuse, no_external_calls
from pipelines.origination import Origination
from pipelines.entry_points import review_scope
from modules.appetite.facility_types import Appetite

# O11 in full; O1, O4 (a declared subset of eleven of the twenty rules), O7 (last
# spread on file), O9 (last grade), O13 (indicative only) in part. Everything
# else not run -- including O2, O3, O5, O6, O10, O12, O14-O17.
PreAssessment = (
    Origination.scoped_by(review_scope, "EP-9")
    | fuse(Appetite)
).last_known_state(
    # Rule 5: it runs on the last-known state of everything AND SAYS SO. Each
    # last-known value carries its own as-at date, and the confidence band is
    # computed from the oldest of them rather than being a constant -- a
    # pre-assessment on a 14-month-old spread and one on a 3-week-old spread are
    # not equally confident and an RM who is told they are will stop reading the
    # band.
    stale_ok=True, carries_as_at=True, confidence_from="oldest_input_age",
)

# Rule 2: it must not create a footprint. No credit enquiry is lodged. Consent
# for an enquiry has usually not been given at this point, so `core.consent`
# gates THE ENTRY POINT ITSELF, not just its data use.
PreAssessment = PreAssessment.gated_by("core.consent", at="entry")

# Rule 5 / profile B: no external calls, enforced rather than intended. The
# framework refuses to compile a pipeline under `no_external_calls` that reaches
# any node declared as making one. That declaration is doc 02's `@breaks_lineage`
# discipline applied to a different axis -- a conspicuous, greppable marker whose
# absence is checkable.
PreAssessment = no_external_calls(PreAssessment)

# Rule 1: it commits nothing and MUST BE INCAPABLE OF APPEARING TO. Its output
# states the three things it did not do, and they are derived from the scope
# matrix rather than written down -- so a future matrix edit that switches O3
# back on updates the disclaimer automatically, and one that does not cannot
# leave a stale "no screening refresh" claim in front of a client.
PreAssessment = PreAssessment.declares_omissions(
    derived_from=review_scope, row="EP-9",
    phrasing="indicative; no bureau enquiry, no screening refresh, no entity re-assessment",
)
