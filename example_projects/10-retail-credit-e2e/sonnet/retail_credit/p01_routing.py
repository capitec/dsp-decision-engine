"""P01 -- Request validation and routing (spec 10 §5.2): entry point 1, product 10 only.

`decision_date` is fixed by the caller (carried on the request, per 00
§7.3 / O-04 -- "today" appears nowhere after this phase) rather than
re-derived here; this module's own job is the structural checks (10 §5.2:
"a term outside the product's range, an amount below the product minimum")
that must fail *before* any client is resolved or any cost is incurred.
A representative subset -- five checks, not the spec's 34 -- because
validation *volume* is not this project's difficulty; that a rejection
here is structurally distinct from a credit decline (10 §5.2: "must never
be recorded as one") is.

`entry_point_code` and `phase_set_id` for this project's served path are
resolved once, outside decider, by `pipeline.build()` calling
`retail_credit.entry_points.resolve_phase_set`/`phase_set_id` before the
request is scored -- see that module's docstring for why (§4.6:
`phase_set_id` is not a per-record computed value in the way a credit
figure is; it is a routing decision made once, at admission, the same way
`decision_id` is assigned once before any logic runs).
"""
from __future__ import annotations

from decider import step


def validate_request(
    requested_amount: float, term_months: int, product_code: int, channel_code: int,
) -> tuple[bool, list[int]]:
    """A structured rejection (10 §5.2), never recorded as a credit decline -- this project's
    outcome vocabulary keeps it out of `decline_reason_codes` entirely (see `pipeline.py`).
    """
    reasons = []
    if requested_amount <= 0:
        reasons.append(9001)
    if term_months <= 0:
        reasons.append(9002)
    if product_code != 10:
        reasons.append(9003)  # this project's slice: product 10 only
    if channel_code <= 0:
        reasons.append(9004)
    return len(reasons) == 0, reasons


validate_request_step = step(validate_request, outputs=("p01_valid", "p01_rejection_reasons"))
