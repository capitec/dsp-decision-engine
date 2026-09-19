"""Stage 5.2 -- consent, the fraud handoff, and the documented bypass.

Two things here that the framework has to make structural rather than
disciplined.

1.  **Proceeding without bureau consent must be impossible, not discouraged.**
    §5.2: "the flow must make it structurally impossible."  Discipline does not
    survive a refactor.  `requires=` on a module is a declared precondition
    checked by the *graph*: `bureau_quality.DataQuality` declares
    `requires=consent_valid_for_bureau_enquiry`, and composing it into a
    pipeline where no upstream module produces that value is a build error, in
    the same class as an unbound input.  It costs one keyword and it makes the
    statutory breach a compile failure.  See FRAMEWORK-DEMANDS #7.

2.  **`unavailable` is not `approve`.**  The fraud verdict has four values and
    the timeout maps to a fourth, not to the first.  Encoding that as a bool
    anywhere loses it permanently.
"""

from __future__ import annotations

from decider2 import Branch, module, param, step
from decider2.missing import Maybe
from decider2.money import Money

FRAUD_APPROVE, FRAUD_REFER, FRAUD_DECLINE, FRAUD_UNAVAILABLE = 1, 2, 3, 4


@step(output="consent_valid_for_bureau_enquiry", produces_precondition=True)
def consent_check(
    consent_records: "Ragged[ConsentRecord, 12]",
    decision_date: "Date",
    channel_code: int,
) -> bool:
    """Bureau-enquiry consent exists, is valid at `decision_date`, and was obtained
    through a channel permitted for that consent type."""
    pass  # absent or expired -> referral 1201, queue 1.  Never a decline.


ConsentCheck = module(consent_check, name="consent")


# ---------------------------------------------------------------------------
# The bypass.  0.9% of calls time out.  Below R15 000, with 24+ months' internal
# tenure and no adverse internal history, a low-risk bypass applies -- with its
# authority reference and the observed latency recorded, and a monitored share.
# ---------------------------------------------------------------------------


@step(output="fraud_bypass_applied")
def low_risk_bypass(
    fraud_verdict_code: int,
    requested_amount: Maybe[Money],
    internal_tenure_months: Maybe[float],
    has_adverse_internal_history: bool,
    bypass_amount_ceiling: Money = param(Money("15000.00"), owner="financial_crime"),
    bypass_tenure_floor: float = param(24.0, ge=0, owner="financial_crime"),
) -> bool:
    """Apply the documented low-risk bypass when the fraud engine is unavailable."""
    pass


@step(output="fraud_handling_path")
def handling_path(fraud_verdict_code: int, fraud_bypass_applied: bool) -> int:
    """Which of the four handling paths was taken. Recorded on every application."""
    pass  # 1 continue | 2 assess-then-force-refer | 3 decline | 4 unavailable-then-bypass-or-refer


FraudDisposition = module(
    low_risk_bypass,
    handling_path,
    name="fraud_disposition",
    taps=["fraud_handling_path", "fraud_bypass_applied", "fraud_response_ms"],
    # A monitored share is an aggregate over a tap column, computed by project
    # 09 from the decision records.  It is deliberately NOT computed here: a
    # per-record module that needs yesterday's bypass rate would be reading
    # state, and a pure step cannot.
)
