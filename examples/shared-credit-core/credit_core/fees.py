"""Statutory fees. The caps are gazetted per year, so they are looked up by `decision_date`."""
from __future__ import annotations

from datetime import date

from decikit import DecisionTableConfig, missing_as, param, step

from credit_core.dates import EffectiveVersions, Version
from credit_core.evidence import cell_id

fee_caps_version_step = EffectiveVersions(family="fee_caps", versions=[
    Version(version_id="fee_caps@2025", effective_from=date(2025, 1, 1), effective_to=date(2026, 1, 1)),
    Version(version_id="fee_caps@2026", effective_from=date(2026, 1, 1)),
])

_CAPS = {"fee_caps@2025": (6_200.0, 65.0), "fee_caps@2026": (6_500.0, 69.0)}  # (initiation, monthly service)


def build_fee_caps_table() -> DecisionTableConfig:
    """The caps in force for `fee_caps_version`: `initiation_fee_cap`, `monthly_service_fee_cap`, `fee_caps_cell_id`."""
    rows = [{"version": v, "initiation": init, "service": service, "cell_id": cell_id("fee_caps", v, "caps")}
            for v, (init, service) in _CAPS.items()]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "fee_caps_table",
        "columns": {"version": "String", "initiation": "Float64", "service": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "eq", "variable": "fee_caps_version", "value_column": "version"},
        "outputs": ["initiation", "service", "cell_id"],
        "default": [None, None, None],
    }).relabel(writes={"initiation": "initiation_fee_cap", "service": "monthly_service_fee_cap",
                       "cell_id": "fee_caps_cell_id"})


def initiation_fee(
    offered_amount: float,
    base_fee: float = param(1050.0, ge=0.0),
    marginal_rate: float = param(0.10, ge=0.0, le=1.0),
    threshold: float = param(10000.0, ge=0.0),
    initiation_fee_cap: float = missing_as(6_500.0),
) -> float:
    """A base fee plus a share of the amount over `threshold`, never above the cap."""
    return min(base_fee + marginal_rate * max(0.0, offered_amount - threshold), initiation_fee_cap)


def initiation_fee_capped(initiation_fee: float, initiation_fee_cap: float = missing_as(6_500.0)) -> bool:
    return initiation_fee >= initiation_fee_cap


def monthly_service_fee(flat_fee: float = param(69.0, ge=0.0), monthly_service_fee_cap: float = missing_as(69.0)) -> float:
    return min(flat_fee, monthly_service_fee_cap)


initiation_fee_step = step(initiation_fee)
initiation_fee_capped_step = step(initiation_fee_capped)
monthly_service_fee_step = step(monthly_service_fee)
