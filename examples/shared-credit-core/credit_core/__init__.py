"""credit_core: the shared credit decision library.

Each capability is a module of decikit steps and plain functions; import the module you need
(`from credit_core import rate_card`). The pipeline in `credit_core.pipeline` composes them into a Flex Loan quote.

Capability                  Module
dates, evidence             credit_core.dates, credit_core.evidence
reason codes, overlays      credit_core.reason_codes, credit_core.adjustments
rounding, instalment, fees  credit_core.rounding, credit_core.instalment, credit_core.fees
rate card, credit life      credit_core.rate_card, credit_core.credit_life
scorecard, calibration      credit_core.scorecard, credit_core.calibration, credit_core.risk_grade
income and affordability    credit_core.income, deductions, expense_norms, obligations, affordability
bureau, eligibility         credit_core.bureau, credit_core.eligibility
appetite, exposure          credit_core.appetite, credit_core.exposure
consent, adverse events     credit_core.consent, credit_core.adverse_events

`credit_core.vocab` holds the codes and names every consumer shares.
"""
from credit_core.adjustments import AdjustmentResult, AdjustmentStack, Effect, Overlay
from credit_core.dates import EffectiveVersions, Version
from credit_core.evidence import cell_id, new_decision_id
from credit_core.reason_codes import Code, ReasonCodes
from credit_core.rounding import round_advance, round_instalment, round_rate

__all__ = [
    "AdjustmentResult", "AdjustmentStack", "Code", "EffectiveVersions", "Effect", "Overlay", "ReasonCodes", "Version",
    "cell_id", "new_decision_id", "round_advance", "round_instalment", "round_rate",
]
