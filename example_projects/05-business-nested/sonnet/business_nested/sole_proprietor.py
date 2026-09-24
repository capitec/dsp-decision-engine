"""Stage -- sole-proprietor regulated affordability (spec 05 §5.2, §5.12 item 7):
"a statutory affordability assessment is required on the natural person behind the
facility, through project 02 ... A statutory fail is a hard fail." Only reached
when `legal_form_code == 1` (sole proprietor).

**Framework/consumption note (see NOTES.md "Gaps in what I consumed").** Project 02
is a product flow, not a library like project 00's `credit_core` -- it has no
uniquely-named importable package, only a top-level `pipeline.py` with a `build()`
function, exactly like this project's own `pipeline.py`. A plain `import pipeline`
here would silently resolve to *this* project's own `pipeline.py` (or whichever
`pipeline.py` sys.path finds first) rather than project 02's, because both projects
use the same entry-point filename by the convention BRIEF.md itself sets ("`pipeline.py`
with `build(...)`"). `_load_affordability_pipeline()` below works around that by
loading project 02's `pipeline.py` from its own file path, under a private module
name, found by locating `assessment/household.py` on `sys.path` -- a file unique to
project 02's own internal package. Project 11 will hit the identical problem
consuming *this* project's `pipeline.py` from inside its own.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import polars as pl

from decider import Engine, frame_step

_AFFORDABILITY_MARKER = Path("assessment") / "household.py"
_affordability_module = None


def _load_affordability_pipeline():
    global _affordability_module
    if _affordability_module is not None:
        return _affordability_module
    for entry in sys.path:
        candidate = Path(entry) / _AFFORDABILITY_MARKER
        if candidate.is_file():
            pipeline_path = Path(entry) / "pipeline.py"
            spec = importlib.util.spec_from_file_location("business_nested._affordability_pipeline_02", pipeline_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _affordability_module = module
            return module
    raise ImportError(
        "project 02 (affordability) not found on sys.path -- add its directory to "
        "PYTHONPATH (see SERVE.md); looked for 'assessment/household.py' on every entry"
    )


_AFFORDABILITY_ENGINE = None


def _affordability_engine():
    global _AFFORDABILITY_ENGINE
    if _AFFORDABILITY_ENGINE is None:
        module = _load_affordability_pipeline()
        _AFFORDABILITY_ENGINE = Engine().bind(module.build())
    return _AFFORDABILITY_ENGINE


def _placeholder_applicant2_account() -> dict:
    """Project 02's own trap (its NOTES.md "Framework friction" 4.1): a solo
    application's applicant2 ragged fields must carry one harmless placeholder
    element, never `[]` or an omitted key, or single-record scoring crashes."""
    return {"account_type_code": 31, "balance": 0.0, "limit": 0.0, "instalment": 0.0,
            "months_in_arrears": 0, "opened_date": date(2020, 1, 1), "closed": True, "is_internal": False}


def build_affordability_request(
    decision_date, principal_entity: dict, business_net_profit_after_drawings: float, proposed_instalment: float,
) -> dict:
    """Maps the sole-proprietor principal entity + the business's own numbers onto
    project 02's request contract (SERVE.md/sample_request.json there documents the
    exact field names this mirrors)."""
    placeholder = _placeholder_applicant2_account()
    return {
        "decision_id": f"sp-{principal_entity.get('entity_key', 'unknown')}",
        "decision_date": decision_date, "product_code": 50, "channel_code": 4, "segment_code": 4,
        "assessment_mode_code": 1, "is_joint_application": False, "risk_grade": 6,
        "applicant1_dependants_count": 0, "applicant1_employment_type_code": 3,  # self-employed
        "applicant1_payslip_income": max(0.0, business_net_profit_after_drawings),
        "applicant1_variable_pay_history": [business_net_profit_after_drawings],
        "applicant1_declared_expenses": {}, "applicant1_statement_expenses": {},
        "applicant1_bureau_accounts": [placeholder], "applicant1_internal_accounts": [placeholder],
        "applicant2_dependants_count": 0, "applicant2_employment_type_code": 0,
        "applicant2_variable_pay_history": [0.0], "applicant2_declared_expenses": {},
        "applicant2_statement_expenses": {}, "applicant2_bureau_accounts": [placeholder],
        "applicant2_internal_accounts": [placeholder],
        "bureau_as_of_date": decision_date, "court_ordered_deductions": 0.0,
        "proposed_instalment": proposed_instalment,
    }


def assess_sole_proprietor(
    legal_form_code: int, decision_date, entity_id: list[int], entity_relationship_type_code: list[int],
    entity_key: list[str], declared_annual_turnover: float, requested_amount: float,
) -> tuple[bool, int, float]:
    """(ran, affordability_verdict_code, max_affordable_instalment) -- `ran=False`
    with a neutral verdict for every legal form except sole proprietor (§5.2)."""
    from business_nested import vocab

    if legal_form_code != 1:
        return False, 0, 0.0
    principal = None
    for i, code in enumerate(entity_relationship_type_code):
        if code == vocab.SOLE_PROPRIETOR_PRINCIPAL:
            principal = {"entity_key": entity_key[i]}
            break
    if principal is None:
        return False, 0, 0.0

    # A crude stand-in for "business net profit after drawings" (spec 05 §5.12 item 7) --
    # this slice has no drawings figure on the request; one-twelfth of declared turnover
    # at a fixed margin assumption is a placeholder, not a real income figure.
    monthly_profit_estimate = declared_annual_turnover * 0.15 / 12.0
    proposed_instalment_estimate = requested_amount * 0.025  # ~2.5%/month, a rough opening probe

    record = build_affordability_request(decision_date, principal, monthly_profit_estimate,
                                          proposed_instalment_estimate)
    result = _affordability_engine().score(record, {})
    return True, result["affordability_verdict_code"], result["max_affordable_instalment"]


@frame_step(
    reads=["legal_form_code", "decision_date", "entity_id", "entity_relationship_type_code", "entity_key",
           "declared_annual_turnover", "requested_amount"],
    writes=["sole_proprietor_assessment_ran", "sole_proprietor_affordability_verdict_code",
            "sole_proprietor_max_affordable_instalment"],
)
def sole_proprietor_step(df: pl.DataFrame) -> pl.DataFrame:
    cols = ["legal_form_code", "decision_date", "entity_id", "entity_relationship_type_code", "entity_key",
            "declared_annual_turnover", "requested_amount"]
    results = [assess_sole_proprietor(**{c: row[c] for c in cols}) for row in df.select(cols).to_dicts()]
    ran, verdict, max_instalment = zip(*results) if results else ((), (), ())
    return df.with_columns(
        pl.Series("sole_proprietor_assessment_ran", list(ran)),
        pl.Series("sole_proprietor_affordability_verdict_code", list(verdict)),
        pl.Series("sole_proprietor_max_affordable_instalment", list(max_instalment)),
    )
