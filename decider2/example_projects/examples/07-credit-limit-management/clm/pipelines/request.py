"""The event-driven path (s5.11). 14 000 client-requested increases a day, p99
under 200 ms, on a programme-run day.

THE ONLY DIFFERENCES FROM THE PROGRAMME ARE DECLARED INPUTS.

  path_code = 2                 -> the exclusion panel's `applies_on` drops X11
                                   and X12 and the cooling-off table gives X09 a
                                   shorter window. No branch, no fork; a key.
  income_basis = declared+deposit -> a different head on the evidence waterfall.
  no Allocation                 -> a client who asks is served. The budget
                                   governs what the Bank pushes out, not what it
                                   grants on request (s5.12).

s5.12 requires that a client requesting in month M on unchanged evidence gets a
proposed limit within max(R500, 2% of L) of what the programme produced. The
tolerance exists only because the EVIDENCE differs; the logic cannot differ,
because `ProposedLimit` below is the same Python object the programme composes.
"""

from decider2 import module, param, pipeline
from clm.affordability.assess import Assess
from clm.offer.construct import ApplyIncrease, Construct
from clm.pipelines.programme import ProposedLimit
from clm.eligibility.panel import Eligibility, Exclusions
from clm.vocabulary import vocabulary


def declared_income_requires_verification(
    declared_gross_income_c: int, deposit_derived_income_c: int,
    tolerance: float = param(0.15, ge=0, le=1.0),
) -> bool:
    """Where fresh declared income is materially higher than deposit-derived
    income, the declared figure is subject to verification before the limit
    changes, and the answer is "approved, subject to confirming your income",
    not a silent decline (s5.11)."""
    pass


def counter_offer_c(requested_amount_c: int, proposed_limit_c: int,
                    current_limit_c: int, minimum_c: int) -> int:
    """min(requested, matrix-and-cap maximum). Where the request exceeds the
    maximum the Bank counter-offers at the maximum if that clears the minimum
    meaningful increase, and otherwise declines with a primary_reason_code from
    the shared taxonomy."""
    pass


RequestHead = module(declared_income_requires_verification, name="request_head")
RequestTail = module(counter_offer_c, name="counter_offer",
                     evidence=["requested_amount_c", "counter_offer_c",
                               "journey_id", "path_code"])

request = pipeline(
    Exclusions | Eligibility | RequestHead | ProposedLimit | RequestTail
    | Construct | ApplyIncrease
).with_vocabulary(vocabulary)

# Realtime invocation bypasses polars entirely (doc 02 s3.5). The artefact set
# is resolved once per generation and held, not per request -- resolving 1 152
# matrix cells and a 60-row overlay register per request would be the whole
# latency budget.
#
#   answer = request.score(
#       account_id=...., current_limit_c=..., requested_amount_c=...,
#       path_code=2,
#       params=params, shared={"decision_date": today}, tables=live,
#   )
#
# Cold start: the kernel is compiled at image build (doc 02 s3.4) and the
# deployment is `sealed`, so the first request after a deploy is not materially
# slower than the thousandth.
#
# THE SAME OVERLAY STACK APPLIES. A cycle dial set for the monthly programme
# applies to client requests made in the same period UNLESS the overlay's
# declared scope excludes them, and `path_code` is one of the declared scope
# keys on every overlay point. Whether a dial is scoped to the programme or to
# the product is a decision the overlay states, because getting it wrong is how
# the two paths start disagreeing (s5.11).
