"""Offer construction, the minimum meaningful increase, and the consent
precondition (s5.9).

Two things happen here that look small and are not.

1. THE MINIMUM MEANINGFUL INCREASE IS APPLIED BEFORE RANKING, AND THE REASON IS
   RECORDED. s5.8's dial table is the whole argument: an 80% dial does not
   shrink the programme, it changes how far down the ranking a fixed budget
   reaches. 41 000 accounts drop below the minimum BECAUSE OF the dial, and
   96 000 more clients get funded as a result. "Reduced below the minimum by
   overlay ADJ-2026-114" and "ranked below the funding line" are different
   answers to the client, different answers to Credit Committee, and only the
   first is reversible by withdrawing an overlay.

   The distinction is computable here and nowhere else, because this is the one
   place both `additional_limit_c` and `additional_limit_unadjusted_c` are in
   scope -- which is the payoff from `shadow` in matrix/assignment.py.

2. CONSENT IS A PRECONDITION IN THE GRAPH, NOT A CHECK IN A DOWNSTREAM SERVICE.
"""

from decider2 import module, param, table
from clm.vocabulary import ALLOCATION_OUTCOMES

MinimumIncrease = table(
    "clm.minimum_increase",
    keys=("product_code",),
    values={"minimum_c": int},
    unit={"minimum_c": "ZAR"}, scale={"minimum_c": 100},
    dense=True, effective_dated=True,
)
# R1 000 Everyday Card, R500 Access Facility. Below this no offer is made: a
# R400 increase costs more in notice and consent handling than it earns, and it
# consumes a cooling-off window.


def meets_minimum(additional_limit_c: int, product_code: int, tables,
                  matrix_min_increment_c: int) -> bool:
    """Above both the product minimum and the cell's own minimum increment."""
    pass


def meets_minimum_unadjusted(additional_limit_unadjusted_c: int, product_code: int,
                             tables, matrix_min_increment_unadjusted_c: int) -> bool:
    """The same test against the shadow. Free, because the shadow already ran."""
    pass


def suppression_reason_code(meets_minimum: bool, meets_minimum_unadjusted: bool,
                            cycle_dial_id: int) -> int:
    """`overlay_suppressed` when the unadjusted value cleared the minimum and the
    adjusted one did not; `below_minimum` when neither did; zero otherwise.

    This is the s10.7 acceptance criterion, and it exists as a column for every
    account rather than as a report someone assembles."""
    pass


def offer_amount_c(funded_amount_c: int, requested_amount_c: int, path_code: int) -> int:
    """The programme offers the matrix-and-cap maximum. A client request is
    min(requested, maximum), with a counter-offer at the maximum where the
    request exceeds it and the counter clears the minimum."""
    pass


def offer_expiry_day(shared, expiry_days: int = param(30, ge=1, le=90)) -> int:
    """30 calendar days from despatch."""
    return shared.decision_date_day + expiry_days


def change_type_code(additional_limit_c: int, path_code: int, is_conditional: bool,
                     is_closure: bool) -> int:
    """One of the nine change types (s6.5), which keys the cooling-off window
    the NEXT cycle will read. The temporal state this project consumes is the
    temporal state it produces."""
    pass


def notice_class_code(change_type_code: int, jurisdiction_code: int, tables) -> int:
    """Immediate, prescribed notice, or consent-required (s6.7)."""
    pass


Construct = module(
    meets_minimum, meets_minimum_unadjusted, suppression_reason_code,
    offer_amount_c, offer_expiry_day, change_type_code, notice_class_code,
    name="offer_construct",
    evidence=[
        "offer_amount_c", "suppression_reason_code", "change_type_code",
        "notice_class_code", "offer_expiry_day",
        # s5.9, mandatory: the unadjusted matrix value on EVERY offer made.
        "proposed_limit_unadjusted_c", "matrix_multiplier_unadjusted",
        "matrix_cell_id", "binding_cap_code",
    ],
)

# s9.2 requires two failure modes to be STRUCTURALLY IMPOSSIBLE rather than
# merely tested for: a limit change with no assessment, and a limit change whose
# consent record post-dates it. `ensures` puts both in the graph. The framework
# cannot police a downstream ledger -- see FRAMEWORK-DEMANDS #13 for the honest
# limit of this -- but it can make the decision record unconstructable without
# them, which is where the 50 000-account regulator sample is drawn from.
ApplyIncrease = module(
    ...,
    name="apply_increase",
    ensures=[
        "affordability_assessment_id != 0",
        "consent_record_id != 0",
        "consent_timestamp < applied_timestamp",
        "affordability_assessed_day <= applied_day",
        "applied_limit_c <= proposed_limit_c",
    ],
    evidence=["offer_id", "consent_record_id", "consent_wording_version",
              "affordability_assessment_id", "despatch_channel_code",
              "despatch_timestamp", "response_code", "response_timestamp"],
)
