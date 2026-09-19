"""Stage 5.3 -- the verdict on the bureau result.

`core.bureau` normalises.  This module owns the verdict, and the verdict's
governing requirement is a NEGATIVE one: a data-quality problem may never
become a decline.  DQ-2 and DQ-3 refer.

The three-states requirement is the interesting part.  §5.3: "no bureau record,
a bureau record showing no accounts, and a bureau enquiry that failed are three
different applicants and must remain three different values through the whole
flow."  With a plain `Optional[float]` all three are `None` by the time they
reach the scorecard, and the scorecard's null bin cannot tell them apart -- so
the thin-file routing rule in §5.4 is unimplementable.  `Maybe[T]` carries an
absence reason alongside the null.  See FRAMEWORK-DEMANDS #6.
"""

from __future__ import annotations

from decider2 import module, param, step
from decider2.collections import Ragged
from decider2.missing import Maybe

DQ_CLEAN, DQ_MINOR, DQ_MATERIAL, DQ_UNUSABLE = 0, 1, 2, 3


@step(output="data_quality_verdict")
def data_quality(
    bureau_subject_count: int,
    bureau_accounts: Ragged["BureauAccount", 80],
    bureau_defects: Ragged["Defect", 32],
) -> int:
    """DQ-0 clean, DQ-1 minor, DQ-2 material, DQ-3 unusable. Never a decline."""
    pass  # >1 subject is DQ-2 unconditionally; automatic merging is prohibited


@step(output="bureau_accounts_truncated")
def account_list_truncated(bureau_accounts: Ragged["BureauAccount", 80]) -> bool:
    """The bureau's account list overflowed the declared capacity of 80.

    `Ragged[..., 80]` declares `on_overflow="flag"`, so an 81st account sets
    this flag and raises DQ-2 rather than raising an exception or silently
    dropping an adverse item.  An overflow is a data-quality FACT, so it must
    produce a value; a framework that can only raise on overflow makes this
    rule unwritable.
    """
    pass


@step(output="bureau_is_stale")
def staleness(
    bureau_as_of_date: "Date",
    decision_date: "Date",
    staleness_tolerance_days: int = param(45, ge=0, le=365, owner="credit_risk_policy"),
) -> bool:
    """Older than the product's tolerance at `decision_date`. Flex Loan 45 days."""
    pass  # the tolerance is per-product and must be settable without touching Consolidation


@step(output="bureau_state")
def bureau_state(
    bureau_subject_count: int,
    bureau_accounts: Ragged["BureauAccount", 80],
    bureau_enquiry_succeeded: bool,
) -> int:
    """NO_HIT / NO_ACCOUNTS / ENQUIRY_FAILED / HIT -- four values, never collapsed."""
    pass


@step(output="enquiry_velocity_60d")
def enquiry_velocity_60d(
    bureau_enquiries: Ragged["Enquiry", 120], decision_date: "Date"
) -> int:
    """Enquiries in the trailing 60 days. Four windows are derived: 30/60/90/365."""
    pass  # a Fold over a Ragged -- record tier, identical in apply() and score()


DataQuality = module(
    data_quality,
    account_list_truncated,
    staleness,
    bureau_state,
    enquiry_velocity_60d,
    name="bureau_quality",
    requires=["consent_valid_for_bureau_enquiry"],  # see fraud_handoff.py
    taps=["data_quality_verdict", "bureau_state", "bureau_is_stale"],
)
