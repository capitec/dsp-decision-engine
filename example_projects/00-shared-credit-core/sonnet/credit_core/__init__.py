"""credit_core: the Bank's shared credit decision library (spec 00; addendum 00-ADDENDUM).

Twenty-two named, versioned capabilities, one module each -- import the
module you need directly (`from credit_core import rate_card`); this
top-level package only re-exports the mechanisms every capability and
every consumer shares.

Capability -> module:
    core.dates          credit_core.dates          (EffectiveDatedSet, EffectiveVersion)
    core.reason_codes    credit_core.reason_codes    (ReasonCode, ReasonCodeRegistry)
    core.adjustments      credit_core.adjustments      (Adjustment, AdjustmentRegister, AdjustmentEffect)
    core.rounding          credit_core.rounding
    core.instalment          credit_core.instalment          (+ inverse: solve_advance_for_instalment)
    core.fees                 credit_core.fees
    core.rate_card              credit_core.rate_card              (generate_flex_loan_card, diff_cards)
    core.scorecard                credit_core.scorecard
    core.calibration                credit_core.calibration
    core.risk_grade                   credit_core.risk_grade
    core.income                         credit_core.income
    core.deductions                       credit_core.deductions
    core.expense_norms                      credit_core.expense_norms
    core.obligations                          credit_core.obligations
    core.affordability                          credit_core.affordability
    core.bureau                                   credit_core.bureau
    core.eligibility                                credit_core.eligibility
    core.appetite                                     credit_core.appetite
    core.exposure                                       credit_core.exposure
    core.consent                                          credit_core.consent
    core.credit_life                                        credit_core.credit_life
    core.adverse_events                                       credit_core.adverse_events

See NOTES.md "What I publish" for a consumer-facing quickstart, and
`credit_core.vocab` for the canonical field names every capability and
every consumer shares unchanged.
"""
from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister, AdjustmentResult
from credit_core.dates import EffectiveDatedSet, EffectiveVersion
from credit_core.evidence import cell_id, new_decision_id
from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry
from credit_core.rounding import round_advance, round_instalment, round_rate

__all__ = [
    "Adjustment", "AdjustmentEffect", "AdjustmentRegister", "AdjustmentResult",
    "EffectiveDatedSet", "EffectiveVersion",
    "cell_id", "new_decision_id",
    "ReasonCode", "ReasonCodeRegistry",
    "round_advance", "round_instalment", "round_rate",
]
