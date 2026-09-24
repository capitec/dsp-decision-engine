"""`core.deductions` -- statutory deductions (spec 00 §6.2).

Tax tables are effective-dated and selected by `decision_date`, never by
today (00 §6.2 "Hard part"): a 2028 replay of a 2026 decision must use the
2026 table. Employment types that are not PAYE-withheld (self-employed,
informal) are not run through the bracket table -- they carry no
withholding to model here, which is a real distinction, not a shortcut.
"""
from __future__ import annotations

from datetime import date

from decider.steps.tables import DecisionTableConfig

from credit_core.dates import EffectiveDatedSet, EffectiveVersion
from credit_core.evidence import cell_id as _cell_id

# employment_type_code: 1 permanent, 2 contract, 3 self-employed, 4 pensioner, 5 social grant, 6 informal.
NOT_PAYE_WITHHELD = {3, 6}

TAX_TABLES = EffectiveDatedSet(
    "tax_table",
    [
        EffectiveVersion("tax@2025", date(2025, 3, 1), date(2026, 3, 1)),
        EffectiveVersion("tax@2026", date(2026, 3, 1), None),
    ],
)
tax_table_version_step = TAX_TABLES.resolver_step(output="tax_table_version")

# Progressive brackets per version: (lo, hi, base_amount, marginal_rate). `lo` starts at 0.0
# (income is never negative) rather than an open edge, so `band_floor` below is always finite;
# `hi` on the last bracket is +inf, not None (see credit_core.rate_card._bands).
_INF = float("inf")
_BRACKETS = {
    "tax@2025": [
        (0.0, 8_500.0, 0.0, 0.0),
        (8_500.0, 20_000.0, 0.0, 0.18),
        (20_000.0, 45_000.0, 2_070.0, 0.26),
        (45_000.0, _INF, 8_570.0, 0.31),
    ],
    "tax@2026": [
        (0.0, 8_900.0, 0.0, 0.0),
        (8_900.0, 21_000.0, 0.0, 0.18),
        (21_000.0, 47_000.0, 2_178.0, 0.26),
        (47_000.0, _INF, 8_938.0, 0.31),
    ],
}


def build_tax_table() -> DecisionTableConfig:
    # Output columns are namespaced (tax_base/tax_rate/tax_band_lo/tax_table_cell_id, not the
    # generic base/rate/lo/cell_id) so this table composes with every other table-backed
    # capability in one flow/dag without a column-name collision (00 §6.9's rate_card and
    # this table would otherwise both write "rate").
    rows = []
    for version, brackets in _BRACKETS.items():
        for i, (lo, hi, base, rate) in enumerate(brackets):
            rows.append({
                "version": version, "lo": lo, "hi": hi, "base": base, "rate": rate,
                "cell_id": _cell_id("tax_table", version, i),
            })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "tax_brackets",
        "columns": {"version": "String", "lo": "Float64", "hi": "Float64", "base": "Float64", "rate": "Float64",
                    "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "tax_table_version", "value_column": "version"},
            {"type": "between", "variable": "gross_monthly_income", "lower_bound_column": "lo",
             "upper_bound_column": "hi", "allow_gaps": True},
        ]},
        "outputs": ["base", "rate", "lo", "cell_id"],
        "default": [0.0, 0.0, 0.0, None],
    }).relabel(writes={"base": "tax_base", "rate": "tax_rate", "lo": "tax_band_lo", "cell_id": "tax_table_cell_id"})


def statutory_deductions(
    gross_monthly_income: float, employment_type_code: float, tax_base: float, tax_rate: float, tax_band_lo: float,
) -> float:
    if employment_type_code in NOT_PAYE_WITHHELD:
        return 0.0
    return round(tax_base + tax_rate * (gross_monthly_income - tax_band_lo), 2)


def net_monthly_income(gross_monthly_income: float, statutory_deductions: float) -> float:
    return round(gross_monthly_income - statutory_deductions, 2)
