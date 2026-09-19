"""Stage 5.1 -- one account list from two sources.

The bureau list (0..80) and the internal list (0..25) overlap, because the
Bank's own facilities are also reported to the bureau -- with a different
identifier, a different account type code, and a lag of up to 60 days.

This is a GROUP-BY over a ragged list that lives inside one record. Doc 02 5
puts joins and group-bys in the frame tier, and that is the right home for a
group-by across records. It is the wrong home for this one: an application is
one row, the accounts have no independent row identity, and pushing the dedup
to polars would mean exploding 0..105 accounts per application into a frame,
grouping, and joining back -- which is exactly the filter/group-by/join-back
rewrite doc 03 8.3 records as having destroyed early exit in the previous
generation.

So `over()` takes a `group=` phase: a record-tier grouping over a bounded
collection, with a declared collapse. FRAMEWORK-DEMANDS #6.
"""

from decider2 import over, policy, rung, step


@step(
    description=(
        "The dedup key: provider identity, account type family, opened date "
        "within a tolerance, and original advance within a percentage or an "
        "absolute amount, whichever is larger."
    ),
)
def dedup_key(
    provider_id: int,
    account_type_family: int,
    opened_date_bucket: int,
    original_advance_cents: int,
    opened_date_tolerance_days: int = policy(45, ge=0, le=365),
    advance_tolerance_pct: float = policy(0.05, ge=0.0, le=1.0),
    advance_tolerance_floor_cents: int = policy(50_000, ge=0),
) -> int:
    pass  # a fuzzy key; the tolerances are parameters because a merge of two genuinely
          # different accounts is one of the three named common defects (spec 9.4)


@rung(
    section="obligations",
    order=80,
    says=(
        "Account {account_reference} appeared on {group_member_count} "
        "source{group_member_count:plural}. The {dedup_source_won:source} "
        "figures were used, being the current ones. "
        "{dedup_discrepancy_note}"
    ),
)
@step(description="A group with an internal member takes the internal balance, instalment and arrears figures.")
def collapse_group(
    group: "Group[Account]",
) -> "Account":
    pass  # internal member wins where present; the bureau's figures are stale by up to 60 days


@step(
    output="dedup_discrepancy",
    description=(
        "Where the two sources disagree materially on instalment or balance, "
        "the discrepancy is recorded. It is not resolved silently."
    ),
)
def dedup_discrepancy(group: "Group[Account]") -> bool:
    pass  # material disagreement between an internal and a bureau member of one group


@step(
    description=(
        "Closure requires either an internal settled status or a bureau closed "
        "status. One source alone downgrades it to a recorded discrepancy and "
        "the account stays in, at its stated instalment."
    ),
)
def closure_established(
    internal_settled: bool | None,
    bureau_closed: bool | None,
) -> bool:
    pass  # NOT (internal_settled or bureau_closed) -- it is an OR over two *present* sources,
          # and an absent source does not vote. Dropping obligations on thin evidence is the
          # optimistic direction, which is the direction that is reckless.


@step(
    description=(
        "A court-ordered deduction in respect of a credit agreement matches a "
        "bureau account. The account is suppressed, because the deduction is "
        "already taken before discretionary income."
    ),
)
def suppressed_by_court_order(
    provider_id: int,
    court_order_credit_agreement_keys: "IdSet",
) -> bool:
    pass  # provider_id in the key set emitted upstream of the evidence cut


AccountList = over(
    "raw_accounts",                       # bureau (0..80) ++ internal (0..25)
    steps=[dedup_key],
    group="dedup_key",
    collapse=collapse_group,
    also=[dedup_discrepancy, closure_established, suppressed_by_court_order],
    emits="accounts",                     # the deduplicated list, 0..80
    annotate=["dedup_key", "dedup_source_won", "dedup_discrepancy", "group_member_count"],
    max_elements=105,
    max_groups=80,
)


# --------------------------------------------------------------------------
# Accounts opened since bureau_as_of_date are invisible here by construction.
# The uplift for that is a NAMED LINE with its own treatment code, never a
# silent addition, so that analysis can exclude it.
# --------------------------------------------------------------------------

@rung(
    section="obligations",
    order=88,
    says=(
        "A policy uplift of {enquiry_velocity_uplift_cents:money} was added for "
        "credit-seeking activity: {enquiry_count_30d} credit enquiries in the "
        "30 days to {bureau_as_of_date:date}, against a threshold of "
        "{enquiry_velocity_threshold}. Accounts opened after "
        "{bureau_as_of_date:date} are not visible on this bureau view."
    ),
)
@step(description="An uplift for credit-seeking in the window the bureau view cannot see.")
def enquiry_velocity_uplift_cents(
    enquiry_count_30d: int,
    enquiry_velocity_threshold: int = policy(2, ge=1, le=20),
    uplift_cents: int = policy(75_000, ge=0),
) -> int:
    pass  # a named line in existing_obligations, with its own treatment code


@step(
    output="bureau_is_stale",
    description=(
        "The bureau view must have been obtained within a short window before "
        "approval -- seven business days for non-mortgage credit and fourteen "
        "for mortgage-secured products."
    ),
)
def bureau_is_stale(
    bureau_as_of_date: "Date",
    decision_date: "Date",
    product_code: int,
    window_business_days: int = policy(7, ge=1, le=60),
    mortgage_window_business_days: int = policy(14, ge=1, le=60),
) -> bool:
    pass  # business days, not calendar days; a stale view produces indeterminate, not fail
