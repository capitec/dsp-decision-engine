"""
Consolidation and Restructure (project 06) — Settlement search and scenario evaluation.

Finds which settlement set + product combination best serves a client
requesting new money or in financial distress.
"""
from __future__ import annotations
from datetime import date
from typing import Optional
from decider import flow, param, missing_as

from consolidation import assess_consolidation


def consolidate(
    client_id: int = param(1, ge=1),
    application_id: int = param(1, ge=1),
    decision_date: date = missing_as(date.today()),
    channel_code: int = param(1, ge=1, le=4),
    requested_amount: float = missing_as(0.0),
    assessment_mode_code: int = param(1, ge=1, le=4),
    client_nominated_settle: list = missing_as([]),
    client_excluded_settle: list = missing_as([]),
    hardship_declared: bool = param(False),

    # Client context from caller
    gross_monthly_income: float = param(15000.0, ge=0.0),
    net_monthly_income: float = param(11000.0, ge=0.0),
    living_expenses: float = param(5000.0, ge=0.0),

    # Obligations inventory (external + internal)
    accounts: list = missing_as([]),

    # Budget and objectives
    scenario_budget: int = param(400, ge=40, le=1000),
    objective_id: int = param(2, ge=1, le=5),
    objective_weights: dict = missing_as({}),

    # Policy interventions (defaults)
    con_int_01_max_accounts: int = param(8, ge=3, le=15),
    con_int_02_months_since_open: int = param(3, ge=0, le=12),
    con_int_03_relief_floor_pct: float = param(0.10, ge=0.0, le=0.30),
    con_int_04_anti_harm_pct: float = param(0.15, ge=0.0, le=0.60),
    con_int_05_term_extension_max: int = param(24, ge=0, le=60),
    con_int_08_new_money_pct: float = param(0.25, ge=0.0, le=0.50),
    con_int_08_new_money_cap: float = param(50000.0, ge=0.0, le=150000.0),
    con_int_09_dsr_ceiling: float = param(0.45, ge=0.30, le=0.60),
    con_int_10_external_min_pct: float = param(0.60, ge=0.0, le=1.0),
    con_int_14_disc_income_floor: float = param(850.0, ge=500.0, le=2500.0),

) -> dict:
    """Consolidation assessment: settle obligations, evaluate scenarios, select best."""

    request = {
        "client_id": client_id,
        "application_id": application_id,
        "decision_date": decision_date,
        "channel_code": channel_code,
        "requested_amount": requested_amount,
        "assessment_mode_code": assessment_mode_code,
        "client_nominated_settle": client_nominated_settle,
        "client_excluded_settle": client_excluded_settle,
        "hardship_declared": hardship_declared,
        "gross_monthly_income": gross_monthly_income,
        "net_monthly_income": net_monthly_income,
        "living_expenses": living_expenses,
        "accounts": accounts,
        "scenario_budget": scenario_budget,
        "objective_id": objective_id,
        "objective_weights": objective_weights,
        "policy_thresholds": {
            "con_int_01_max_accounts": con_int_01_max_accounts,
            "con_int_02_months_since_open": con_int_02_months_since_open,
            "con_int_03_relief_floor_pct": con_int_03_relief_floor_pct,
            "con_int_04_anti_harm_pct": con_int_04_anti_harm_pct,
            "con_int_05_term_extension_max": con_int_05_term_extension_max,
            "con_int_08_new_money_pct": con_int_08_new_money_pct,
            "con_int_08_new_money_cap": con_int_08_new_money_cap,
            "con_int_09_dsr_ceiling": con_int_09_dsr_ceiling,
            "con_int_10_external_min_pct": con_int_10_external_min_pct,
            "con_int_14_disc_income_floor": con_int_14_disc_income_floor,
        }
    }

    return assess_consolidation(request)


def build():
    """Build the consolidation and restructure pipeline."""
    return flow(consolidate, name="consolidation_assessment")
