"""The sixteen hard exclusions (s5.2), as a panel.

s5.2 states a tension and refuses to resolve it in the framework's favour:
"complete attribution, and no scoring, affordability or bureau work performed
for an excluded account." A short-circuiting `Branch` chain loses the reason
set. Evaluating everything wastes 65% of the book's bureau and affordability
work.

The answer is that the two halves belong to different tiers.

  * Complete attribution is RECORD tier. Sixteen scalar predicates over columns
    already in memory is sixteen compares -- about 40 ms over 4.1 M rows in one
    kernel. There is nothing to save by skipping them, so nothing is skipped
    and every exclusion that applied is recorded.

  * Work avoidance is FRAME tier. `Partition` (pipelines/programme.py) splits
    the book on `is_considered` and only the taken side runs scoring,
    affordability and the bureau join. The skipped side is schema-completed and
    unioned back, so all 4.1 M accounts get an output record -- "we did consider
    your account" is itself an answer (s7).

Doc 03 has no way to say this. `Branch` short-circuits inside a record, which
is the wrong granularity: what needs skipping is a bureau join and an
affordability assessment, both of which are stages, not steps.
"""

from decider2 import module, panel, param, table
from clm.vocabulary import EXCLUSIONS

# Each member is an ordinary module writing one bool named `excluded`. A member
# is its own scope, exactly like a branch arm (doc 03 s3), so sixteen modules
# may all write `excluded` without collision.


def in_arrears_now(cycles_past_due_current: int) -> bool:
    """X01 - one or more cycles past due at the snapshot."""
    return cycles_past_due_current >= 1


def arrears_within_6m(worst_delinquency_6m: int) -> bool:
    """X02 - any cycle at least one past due in the last six."""
    return worst_delinquency_6m >= 1


def debt_review(debt_review_status_code: int) -> bool:
    """X03 - debt review or administration order, any status but terminated."""
    pass


def insolvency(insolvency_status_code: int) -> bool:
    """X04 - any active sequestration or insolvency status."""
    pass


def deceased_or_estate(marker_deceased: bool, marker_estate: bool) -> bool:
    """X05 - deceased or estate marker on the client."""
    return marker_deceased or marker_estate


def fraud_marker(marker_fraud_account: bool, marker_fraud_client: bool) -> bool:
    """X06 - fraud marker on the account or the client."""
    return marker_fraud_account or marker_fraud_client


def account_dispute(open_dispute_count: int) -> bool:
    """X07 - an open dispute on any transaction."""
    return open_dispute_count > 0


def dormant(current_balance_c: int, months_since_last_transaction: int,
            window: int = param(6, ge=1, le=24)) -> bool:
    """X08 - zero balance and no transaction for the dormancy window."""
    return current_balance_c == 0 and months_since_last_transaction >= window


def cooling_off(months_since_last_limit_change: int,
                cooling_off_window_months: int) -> bool:
    """X09 - inside the window set by the last change type, on this path (s6.5)."""
    return months_since_last_limit_change < cooling_off_window_months


def inflight_application(days_since_undecided_application: int,
                         window_days: int = param(30, ge=0, le=180)) -> bool:
    """X10 - an undecided credit application elsewhere in the Bank."""
    pass


def declined_offer_suppression(months_since_declined_offer: int, path_code: int,
                               tables) -> bool:
    """X11 - inside the declined-offer suppression window. Does not apply on the
    request path: a client who asks has withdrawn their own decline (s5.11)."""
    pass  # window from cooling_off_windows keyed on path_code


def no_increase_consent(has_automatic_increase_consent: bool, path_code: int) -> bool:
    """X12 - no standing agreement to receive increases, or it was withdrawn.

    12.4% of the book. These are NOT declined offers and must never be reported
    as suppressed marketing (s5.2). Does not apply on the request path.
    """
    pass


def staff_account(marker_staff: bool, marker_related_party: bool) -> bool:
    """X13 - staff or related-party marker."""
    return marker_staff or marker_related_party


def too_young(months_on_book: int, minimum: int = param(6, ge=1, le=24)) -> bool:
    """X14 - fewer than six months on book, which is also the scoring minimum."""
    return months_on_book < minimum


def at_product_maximum(current_limit_c: int, product_code: int, tables) -> bool:
    """X15 - already at R300 000 (card) or R150 000 (facility)."""
    pass


def closed_or_arrangement(account_status_code: int, treatment_state_code: int) -> bool:
    """X16 - closed, blocked, or under a project-08 payment arrangement."""
    pass


Exclusions = panel(
    "exclusions",
    members=[in_arrears_now, arrears_within_6m, debt_review, insolvency,
             deceased_or_estate, fraud_marker, account_dispute, dormant,
             cooling_off, inflight_application, declined_offer_suppression,
             no_increase_consent, staff_account, too_young, at_product_maximum,
             closed_or_arrangement],
    reduce="any",
    codes=EXCLUSIONS,
    writes={"value": "is_excluded",
            "fired": "exclusion_codes",        # the COMPLETE set, always
            "primary": "primary_reason_code"},  # ranked by the registered severity
    evidence=["*"],                             # every member's bool survives to the record
    applies_on="path_code",                     # members may declare a path scope
)


def is_considered(is_excluded: bool) -> bool:
    """The `Partition` predicate. Named separately so the partition reads as
    policy rather than as a negation buried in a frame expression."""
    return not is_excluded


Eligibility = module(is_considered, name="eligibility")
