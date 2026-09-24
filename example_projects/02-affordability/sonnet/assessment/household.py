"""Stage 1 -- applicant and household framing (spec 02 §5.1).

A single application assesses one applicant; a joint application assesses
two **as one household**. 00's `core.income` and `core.deductions` units
each take one applicant's evidence; this module runs each unit twice --
once per applicant, `.relabel()`-ed onto that applicant's own fields
(00 NOTES.md "What worked well": exactly the `.named()`/`.relabel()`
composition pattern) -- and combines the two results into the household
figures the rest of the assessment reads under the library's own generic
names (`gross_monthly_income`, `statutory_deductions`, ...), so
`core.expense_norms` and `core.affordability` downstream need no relabelling
at all: they already read a household, they just don't know it.

A solo application runs applicant 2's income/deductions sub-units against
entirely-missing fields, which resolve to `NONE_ESTABLISHED`/zero and
therefore contribute nothing to the household sum -- no `branch` on
`is_joint_application` is needed for that arithmetic. Only the household
*only* concerns (dependants, expense consolidation, the account-list merge,
and "can we assess on one applicant's income alone") are decided here.
"""
from __future__ import annotations

from decider import flow, frame_step, missing_as, param, step

import polars as pl

from credit_core import deductions, income

# Expense category classification (§5.1: "the classification is a parameter, because
# Credit Risk Policy will argue about food"). Working depth: 4 shared + 3 personal
# categories, not the spec's full 8-14 -- the mechanism (a parameter picks higher-of vs.
# sum) is what this proves; `core.expense_norms` (00) already carries the norm-table depth.
SHARED_EXPENSE_CATEGORIES = ("accommodation", "water_electricity", "food", "insurance")
PERSONAL_EXPENSE_CATEGORIES = ("transport", "medical", "communication")


def dependants_count(
    applicant1_dependants_count: int = missing_as(0),
    applicant2_dependants_count: int = missing_as(0),
) -> int:
    """§5.1: where the two applicants declare different counts, the higher is used."""
    return max(applicant1_dependants_count or 0, applicant2_dependants_count or 0)


def _consolidate(cat1: dict | None, cat2: dict | None) -> float:
    cat1 = cat1 or {}
    cat2 = cat2 or {}
    shared = sum(max(cat1.get(c, 0.0) or 0.0, cat2.get(c, 0.0) or 0.0) for c in SHARED_EXPENSE_CATEGORIES)
    personal = sum((cat1.get(c, 0.0) or 0.0) + (cat2.get(c, 0.0) or 0.0) for c in PERSONAL_EXPENSE_CATEGORIES)
    return round(shared + personal, 2)


@frame_step(reads=["applicant1_declared_expenses", "applicant2_declared_expenses"],
            writes=["declared_living_expenses"])
def consolidate_declared_expenses(df: pl.DataFrame) -> pl.DataFrame:
    """§5.1: per category, the higher of the two declarations for shared categories, the
    sum for personal ones. Declared expenses are never absent (the questionnaire is
    mandatory), so this always produces a figure -- zero across all categories is a real,
    recordable "non-declaration" (§5.4), not handled further at this working depth."""
    values = [
        _consolidate(r["applicant1_declared_expenses"], r["applicant2_declared_expenses"])
        for r in df.select("applicant1_declared_expenses", "applicant2_declared_expenses").to_dicts()
    ]
    return df.with_columns(pl.Series("declared_living_expenses", values))


@frame_step(reads=["applicant1_statement_expenses", "applicant2_statement_expenses"],
            writes=["statement_living_expenses"])
def consolidate_statement_expenses(df: pl.DataFrame) -> pl.DataFrame:
    """As above, but statement coverage is frequently entirely absent, and absent must
    stay distinct from a verified zero (§5.4 basis B) -- `None`, not `0.0`, when neither
    applicant has any statement-derived figure at all."""
    rows = df.select("applicant1_statement_expenses", "applicant2_statement_expenses").to_dicts()
    values = []
    for r in rows:
        a1, a2 = r["applicant1_statement_expenses"], r["applicant2_statement_expenses"]
        values.append(None if not a1 and not a2 else _consolidate(a1, a2))
    return df.with_columns(pl.Series("statement_living_expenses", values))


def _applicant_income_unit(prefix: str):
    """00's `core.income` unit, relabelled onto one applicant's fields (§7.5-style reuse:
    the same functions, run twice, never forked)."""
    reads_map = {
        "declared_income": f"{prefix}_declared_income",
        "payslip_income": f"{prefix}_payslip_income",
        "statement_income": f"{prefix}_statement_income",
        "statement_confidence": f"{prefix}_statement_confidence",
        "employer_confirmed_income": f"{prefix}_employer_confirmed_income",
        "bureau_estimated_income": f"{prefix}_bureau_estimated_income",
        "variable_pay_history": f"{prefix}_variable_pay_history",
    }
    writes_map = {
        "gross_monthly_income": f"{prefix}_gross_monthly_income",
        "income_source_code": f"{prefix}_income_source_code",
        "income_verification_tier": f"{prefix}_income_verification_tier",
        "income_haircut_applied": f"{prefix}_income_haircut_applied",
        "income_variability_ratio": f"{prefix}_income_variability_ratio",
    }
    unit = flow(
        income.gross_monthly_income, income.income_source_code, income.income_verification_tier,
        income.income_haircut_applied, income.income_variability_ratio,
        name=f"{prefix}_income",
    )
    return unit.relabel(reads=reads_map, writes=writes_map)


applicant1_income_unit = _applicant_income_unit("applicant1")
applicant2_income_unit = _applicant_income_unit("applicant2")


def _applicant_deductions_unit(prefix: str):
    """00's `core.deductions` unit (tax table + statutory deductions), per applicant --
    tax is individual, so the household is never taxed on its combined income at once
    (that would push a joint household into a bracket neither applicant is actually in)."""
    reads_map = {"gross_monthly_income": f"{prefix}_gross_monthly_income",
                 "employment_type_code": f"{prefix}_employment_type_code"}
    writes_map = {"statutory_deductions": f"{prefix}_statutory_deductions",
                  "net_monthly_income": f"{prefix}_net_monthly_income",
                  "tax_table_version": f"{prefix}_tax_table_version",
                  "tax_base": f"{prefix}_tax_base", "tax_rate": f"{prefix}_tax_rate",
                  "tax_band_lo": f"{prefix}_tax_band_lo", "tax_table_cell_id": f"{prefix}_tax_table_cell_id"}
    unit = flow(
        deductions.tax_table_version_step, deductions.build_tax_table(),
        deductions.statutory_deductions, deductions.net_monthly_income,
        name=f"{prefix}_deductions",
    )
    return unit.relabel(reads=reads_map, writes=writes_map)


applicant1_deductions_unit = _applicant_deductions_unit("applicant1")
applicant2_deductions_unit = _applicant_deductions_unit("applicant2")


def household_gross_monthly_income(
    applicant1_gross_monthly_income: float, applicant2_gross_monthly_income: float = missing_as(0.0),
) -> float:
    """§5.1: incomes are summed after their own individual haircuts."""
    return round(applicant1_gross_monthly_income + (applicant2_gross_monthly_income or 0.0), 2)


def household_income_source_code(
    applicant1_gross_monthly_income: float, applicant1_income_source_code: int,
    applicant2_gross_monthly_income: float = missing_as(0.0), applicant2_income_source_code: int = missing_as(0),
) -> int:
    """§5.2.5: "`income_source_code` records the tier of the largest source" -- at household
    level, the largest applicant."""
    if (applicant2_gross_monthly_income or 0.0) > applicant1_gross_monthly_income:
        return applicant2_income_source_code
    return applicant1_income_source_code


def household_income_verification_tier(
    applicant1_gross_monthly_income: float, applicant1_income_verification_tier: int,
    applicant2_gross_monthly_income: float = missing_as(0.0), applicant2_income_verification_tier: int = missing_as(0),
    minor_contribution_threshold: float = param(0.10, ge=0.0, le=1.0),
) -> int:
    """§5.2.5: the weakest tier among applicants contributing more than 10% of the
    household total. A minor side income at a weak tier does not drag the reported tier
    down, and that must stay visible."""
    total = applicant1_gross_monthly_income + (applicant2_gross_monthly_income or 0.0)
    if total <= 0:
        return max(applicant1_income_verification_tier, applicant2_income_verification_tier or 0)
    tiers = []
    if applicant1_gross_monthly_income / total > minor_contribution_threshold:
        tiers.append(applicant1_income_verification_tier)
    if (applicant2_gross_monthly_income or 0.0) / total > minor_contribution_threshold:
        tiers.append(applicant2_income_verification_tier or 0)
    return max(tiers) if tiers else max(applicant1_income_verification_tier, applicant2_income_verification_tier or 0)


def household_income_haircut_applied(
    applicant1_gross_monthly_income: float, applicant1_income_haircut_applied: float,
    applicant2_gross_monthly_income: float = missing_as(0.0), applicant2_income_haircut_applied: float = missing_as(0.0),
) -> float:
    """§5.2.5: "one minus post-haircut over pre-haircut", blended across the household."""
    a2_gross = applicant2_gross_monthly_income or 0.0
    a2_haircut = applicant2_income_haircut_applied or 0.0
    pre1 = applicant1_gross_monthly_income / (1.0 - applicant1_income_haircut_applied) \
        if applicant1_income_haircut_applied < 1.0 else applicant1_gross_monthly_income
    pre2 = a2_gross / (1.0 - a2_haircut) if a2_haircut < 1.0 else a2_gross
    pre_total = pre1 + pre2
    if pre_total <= 0:
        return 0.0
    post_total = applicant1_gross_monthly_income + a2_gross
    return round(1.0 - post_total / pre_total, 4)


def household_income_variability_ratio(
    applicant1_income_variability_ratio: float, applicant2_income_variability_ratio: float = missing_as(0.0),
) -> float:
    """The more variable of the two applicants' incomes -- the conservative pick."""
    return max(applicant1_income_variability_ratio, applicant2_income_variability_ratio or 0.0)


def statutory_deductions(
    applicant1_statutory_deductions: float, applicant2_statutory_deductions: float = missing_as(0.0),
) -> float:
    return round(applicant1_statutory_deductions + (applicant2_statutory_deductions or 0.0), 2)


def net_monthly_income(
    applicant1_net_monthly_income: float, applicant2_net_monthly_income: float = missing_as(0.0),
) -> float:
    return round(applicant1_net_monthly_income + (applicant2_net_monthly_income or 0.0), 2)


def applicant_income_evidence_gap(
    is_joint_application: bool,
    applicant1_income_source_code: int, applicant2_income_source_code: int = missing_as(0),
) -> int:
    """§5.1: "the household is not assessed on the other's income alone" -- naming which
    applicant's income could not be established at all. 0 = none, 1/2 = which one, 3 = both.
    Checked by evidence (tier `NONE_ESTABLISHED`), never inferred from a zero amount, which
    is a real distinct case (a verified zero income is tier-established, not missing)."""
    a1_missing = applicant1_income_source_code == income.NONE_ESTABLISHED
    a2_missing = bool(is_joint_application) and applicant2_income_source_code == income.NONE_ESTABLISHED
    if a1_missing and a2_missing:
        return 3
    if a1_missing:
        return 1
    if a2_missing:
        return 2
    return 0


_DEDUP_ROUND = 2


def _dedup_key(account: dict) -> tuple:
    return (account.get("account_type_code"), account.get("opened_date"), round(account.get("balance") or 0.0, _DEDUP_ROUND))


def _dedup_within(accounts: list[dict]) -> list[dict]:
    """§5.1: "A joint account reported on both bureau profiles and counted twice is the
    commonest joint-application defect" -- every account counts once, by identity, not by
    which applicant it arrived under."""
    seen: set = set()
    out = []
    for a in accounts:
        key = _dedup_key(a)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


# assessment_mode_code (mirrored from assessment.modes to avoid a circular import).
_SCENARIO_MODE = 4


def _shape_for_mode(accounts: list[dict], assessment_mode_code: int) -> list[dict]:
    """§5.5.2: `EXCLUDE_ON_QUOTE` "only in scenario mode... never in new-application mode".
    00's `core.obligations` turns a live settlement quote into an imputed obligation
    unconditionally (it has no mode concept -- that is 02's to add); outside scenario mode
    the quote is stripped before the shared unit ever sees the account, so its ordinary
    stated/imputed treatment applies instead. See NOTES.md "Gaps in what I consumed"."""
    if assessment_mode_code == _SCENARIO_MODE:
        return accounts
    return [{**a, "settlement_quote": None} for a in accounts]


# Account types the treatment matrix cannot treat mechanically (§5.5.2 `REFER`). Working
# depth: one illustrative type, not a survey of the ~45-type matrix -- the mechanism (an
# unrecognised type forces `indeterminate`, never a silent zero) is what this proves.
REFER_ACCOUNT_TYPES = frozenset({99})

_OBLIGATIONS_OUTPUTS = (
    "existing_obligations", "obligations_internal", "obligations_external", "worst_arrears_months",
    "accounts_in_arrears_count", "total_exposure", "revolving_utilisation",
    "obligation_account_type_codes", "obligation_treatment_codes", "obligation_monthly_amounts",
    "obligation_is_internal",
)


@frame_step(
    reads=["applicant1_bureau_accounts", "applicant2_bureau_accounts",
           "applicant1_internal_accounts", "applicant2_internal_accounts", "assessment_mode_code"],
    writes=[*_OBLIGATIONS_OUTPUTS, "has_refer_account"],
)
def household_obligations(df: pl.DataFrame) -> pl.DataFrame:
    """Stage 5 (§5.5): merges the two applicants' bureau and internal account lists into
    one household list (dedup by identity, §5.1), shapes it for the assessment mode
    (`EXCLUDE_ON_QUOTE` only in scenario mode, §5.5.2), then hands it to 00's own
    per-account arithmetic (`credit_core.obligations._process`) so the treatment matrix,
    the ten behaviours and the scalar-plus-annotation shape are never re-derived here.

    This calls `_process` directly rather than composing the public `obligations.obligations`
    frame_step as a second pipeline member, because the natural design -- write a merged
    `bureau_accounts`/`internal_accounts` *column* here, let `obligations.obligations` read
    it next -- crashes: a `frame_step` output that is a `list[struct]` and then has to flow
    into a *second* `frame_step` fails at the same numpy/polars boundary 00 NOTES.md
    "Framework friction" 4.2 documents for a terminal output, one hop earlier than 00 hit it
    (`ValueError: cannot parse numpy data type dtype('O') into Polars data type`). Calling the
    per-record function directly, inside the one frame_step that already holds the merged
    Python list, never materialises that column at all. See NOTES.md."""
    from credit_core.obligations import _process as _obligations_process

    rows = df.select(
        "applicant1_bureau_accounts", "applicant2_bureau_accounts",
        "applicant1_internal_accounts", "applicant2_internal_accounts", "assessment_mode_code",
    ).to_dicts()
    results = []
    for r in rows:
        mode = r["assessment_mode_code"]
        bureau = _dedup_within((r["applicant1_bureau_accounts"] or []) + (r["applicant2_bureau_accounts"] or []))
        internal = _dedup_within((r["applicant1_internal_accounts"] or []) + (r["applicant2_internal_accounts"] or []))
        bureau = _shape_for_mode(bureau, mode)
        internal = _shape_for_mode(internal, mode)
        out = _obligations_process(bureau, internal)
        out["has_refer_account"] = any(a.get("account_type_code") in REFER_ACCOUNT_TYPES for a in bureau + internal)
        results.append(out)
    return df.with_columns(pl.DataFrame(results))


household_gross_monthly_income_step = step(household_gross_monthly_income, output="gross_monthly_income")
household_income_source_code_step = step(household_income_source_code, output="income_source_code")
household_income_verification_tier_step = step(household_income_verification_tier, output="income_verification_tier")
household_income_haircut_applied_step = step(household_income_haircut_applied, output="income_haircut_applied")
household_income_variability_ratio_step = step(household_income_variability_ratio, output="income_variability_ratio")
