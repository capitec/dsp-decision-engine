"""Statutory deductions: tax on a progressive bracket table selected by `decision_date`."""
from __future__ import annotations

from datetime import date

from decider import DecisionTableConfig

from credit_core.dates import EffectiveDatedSet, EffectiveVersion
from credit_core.evidence import cell_id

# employment_type_code: 1 permanent, 2 contract, 3 self-employed, 4 pensioner, 5 social grant, 6 informal.
# Self-employed and informal earners have no tax withheld at source, so the table does not apply to them.
NOT_PAYE_WITHHELD = {3, 6}

TAX_TABLES = EffectiveDatedSet("tax_table", [
    EffectiveVersion("tax@2025", date(2025, 3, 1), date(2026, 3, 1)),
    EffectiveVersion("tax@2026", date(2026, 3, 1)),
])
tax_table_version_step = TAX_TABLES.resolver_step(output="tax_table_version")

# (bracket floor, tax on the floor, marginal rate) per version.
_BRACKETS = {
    "tax@2025": [(0.0, 0.0, 0.0), (8_500.0, 0.0, 0.18), (20_000.0, 2_070.0, 0.26), (45_000.0, 8_570.0, 0.31)],
    "tax@2026": [(0.0, 0.0, 0.0), (8_900.0, 0.0, 0.18), (21_000.0, 2_178.0, 0.26), (47_000.0, 8_938.0, 0.31)],
}


def build_tax_table() -> DecisionTableConfig:
    """Writes `tax_base`, `tax_rate`, `tax_band_lo` and `tax_table_cell_id` for the record's income and tax version."""
    rows = [{"version": version, "lo": lo, "base": base, "rate": rate, "cell_id": cell_id("tax_table", version, i)}
            for version, brackets in _BRACKETS.items() for i, (lo, base, rate) in enumerate(brackets)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "tax_brackets",
        "columns": {"version": "String", "lo": "Float64", "base": "Float64", "rate": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "tax_table_version", "value_column": "version"},
            {"type": "between", "variable": "gross_monthly_income", "lower_bound_column": "lo"},
        ]},
        "outputs": ["base", "rate", "lo", "cell_id"],
        "default": [0.0, 0.0, 0.0, None],
    }).relabel(writes={"base": "tax_base", "rate": "tax_rate", "lo": "tax_band_lo", "cell_id": "tax_table_cell_id"})


def statutory_deductions(
    gross_monthly_income: float, employment_type_code: int, tax_base: float, tax_rate: float, tax_band_lo: float,
) -> float:
    if employment_type_code in NOT_PAYE_WITHHELD:
        return 0.0
    return round(tax_base + tax_rate * (gross_monthly_income - tax_band_lo), 2)


def net_monthly_income(gross_monthly_income: float, statutory_deductions: float) -> float:
    return round(gross_monthly_income - statutory_deductions, 2)
