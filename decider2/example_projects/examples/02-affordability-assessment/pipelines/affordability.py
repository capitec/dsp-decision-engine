"""The affordability assessment. One pipeline. Five consumers. Four modes.

Nothing in this file is mode-specific and nothing in this file is
consumer-specific. Modes are applied at the call site with `.under(...)`
(policy/modes.py); consumers set local params and choose an *answer shape*.

Read this file top to bottom and you have the regulation's own order:
framing, income, statutory deductions, living expenses, existing obligations,
discretionary income and capacity, verdict. That ordering is load-bearing --
the ombud reads the ladder against the regulation line by line (spec 5.6.1) --
so it lives in the `|` expression where it is visible, and nowhere else.
"""

from decider2 import NON_INCREASING, cut, fuse, monotone_in
from decider2.dating import decision_date

from modules.capacity import Capacity
from modules.deductions import Deductions
from modules.expenses import Expenses
from modules.framing import Framing
from modules.income import Income
from modules.obligations.annotate import Obligations
from modules.verdict import Verdicts
from vocabulary import CREDIT_CORE

# --------------------------------------------------------------------------
# The evidence cut.
#
# Project 06 calls this assessment up to 400 times for one application, varying
# only the obligation set. `cut()` declares ONE boundary; the framework derives
# `hold()`/`resume()` from it. There is no second entry point to drift.
#
# `resumes=` is the *only* thing a resume may change. The framework checks,
# from static lineage and without running anything, that no step upstream of the
# cut reads any name in `resumes` -- if one did, a held prefix would be stale
# and 400 scenarios would silently share a wrong income figure. That check is
# the reason `Framing` does its applicant-level dedup on identity rather than on
# the account list (see modules/framing/__init__.py).
# --------------------------------------------------------------------------
evidence = cut(
    "evidence",
    carries=[
        "gross_monthly_income_cents",
        "net_monthly_income_cents",
        "living_expenses_cents",
        "living_expenses_unadjusted_cents",
        "court_ordered_deductions_cents",
        "income_verification_tier",
        "income_haircut_applied",
        "income_variability_ratio",
        "expense_basis_code",
        "dependants_count",
        "household_size",
        "evidence_sufficiency_code",
    ],
    resumes=[
        "accounts",                    # the ragged 0..80 list
        "settlement_quotes",
        "proposed_instalment_cents",
    ],
)

# --------------------------------------------------------------------------
# The pipeline.
# --------------------------------------------------------------------------
Assessment = (
    Framing
    | Income
    | Deductions
    | Expenses
    | evidence
    | Obligations
    | fuse(Capacity | Verdicts)       # hot: 400x per application, two cheap modules
).with_vocabulary(CREDIT_CORE)

# --------------------------------------------------------------------------
# Declared properties. These are not comments; each generates a test.
# --------------------------------------------------------------------------

# Project 03's binary search over loan amount has no valid stopping condition
# without this. `edges_from` seeds the search corpus from the DECLARED edges of
# the named artefacts -- band floors, grade/product cells, dependant cells,
# the tolerance band -- rather than from random draws, because the one concrete
# requirement doc 03 1.2 places on a corpus is that it contain boundary values
# and random draws never find them.
Assessment = monotone_in(
    Assessment,
    "proposed_instalment_cents",
    direction=NON_INCREASING,
    of="affordability_verdict_code",
    edges_from=[
        "statutory_expense_norms",
        "internal_expense_norms",
        "buffer_grid",
        "residual_floor",
        "verdict_tolerance_pct",
    ],
)

# The scalar obligation figure and the 0..80-row annotation are one computation
# (spec 5.5.3, acceptance 10). This asserts the caller's choice is a write-back
# choice and not a compute choice.
Assessment = Assessment.assert_materialisation_neutral("accounts_annotated")

# Spec 5.7.2(3) / acceptance 4. A fourth rung on the equivalence ladder:
#   interpreted == stepped == fused == held-and-resumed.
Assessment = Assessment.assert_cut_equivalent(evidence)

# --------------------------------------------------------------------------
# `decision_date` is a reserved, frame-only pipeline input. No step may produce
# it, so nothing can overwrite it mid-flow, and every `dated_table` in the tree
# resolves against it. Omit it from the input schema and the build fails:
#
#   build error: 9 effective-dated artefacts require `decision_date`, which is
#   not in schemas/affordability_input.json.
#     statutory_expense_norms   (modules/expenses/norms.py:41)
#     internal_expense_norms    (modules/expenses/norms.py:78)
#     paye_brackets             (modules/deductions/tax.py:23)
#     uif_ceiling               (modules/deductions/statutory.py:18)
#     obligation_treatment      (modules/obligations/treatments.py:96)
#     income_haircuts           (modules/income/haircuts.py:29)
#     minimum_evidence_tier     (modules/income/waterfall.py:57)
#     buffer_grid               (modules/capacity/__init__.py:44)
#     residual_floor            (modules/capacity/__init__.py:52)
#   There is no `.latest` and no `.today`.
# --------------------------------------------------------------------------
assert decision_date in Assessment.schema().inputs
