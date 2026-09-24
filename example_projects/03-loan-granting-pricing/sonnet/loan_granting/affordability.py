"""Stage 5.6 -- affordability, consumed from project 02 (spec 03 §5.6).

"This project does not compute affordability. It consumes project 02's
published assessment" -- so this module is an *adapter*: it maps 03's
application fields onto 02's input contract, calls 02's pipeline once per
application, and republishes the five fields 03 §5.6 names
(`max_affordable_instalment`, `discretionary_income`,
`affordability_verdict_code`, `living_expenses`, `existing_obligations`)
plus every table version 02 resolved, unchanged.

**Loading project 02's `pipeline.py` under project 03's own `pipeline.py`.**
Both projects' entry-point module is, by this repo's own convention
(BRIEF: "`pipeline.py` with `build(...)`"), a file literally named
`pipeline.py`. A plain `import pipeline` from inside *this* project's
`pipeline.py`, while it is itself mid-import under that exact module name,
self-imports (Python caches modules by name, not by path) rather than
reaching project 02's file -- there is no supported way around this with
two sibling projects sharing the same entry-point filename on one
`PYTHONPATH`. Worked around with `importlib`, loading 02's file under a
distinct module name (`affordability_pipeline`), located via the
`assessment` package's own `__file__` (02's package, unique on
`PYTHONPATH`, unambiguous to import directly) rather than scanning
`sys.path`. See NOTES.md "Framework friction" -- 06, which spec 03 §11
item 11 says wants this project's solve, will hit the same collision the
moment it also wants to import 02 by name.

**Why a single call, not one per solve candidate.** `max_affordable_instalment`
depends on income, expenses and obligations -- none of which depend on the
*proposed* instalment -- so it is a ceiling the solve compares candidates
against locally (spec 03 §5.8 preconditions list it as an input, not
something the solve re-derives). Called with no `proposed_instalment`
(project 02's shape (b), "capacity only"), which is the shape that yields a
verdict of `pass`/`indeterminate`, or `fail` only when capacity itself is
zero -- see NOTES.md "Spec problems" for why `marginal` is effectively
unreachable through this shape and what that implies for §5.6's "marginal
restricts offers to 24 months" clause.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

import assessment  # project 02's package -- unique on PYTHONPATH, safe to import directly.


def _load_affordability_pipeline():
    project_dir = Path(assessment.__file__).resolve().parent.parent
    pipeline_path = project_dir / "pipeline.py"
    spec = importlib.util.spec_from_file_location("affordability_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["affordability_pipeline"] = module
    spec.loader.exec_module(module)
    return module


affordability_pipeline = _load_affordability_pipeline()

_EMPTY_ACCOUNT = {"account_type_code": 31, "balance": 0.0, "limit": 0.0, "instalment": 0.0,
                   "months_in_arrears": 0, "opened_date": "2020-01-01", "closed": True, "is_internal": False}


def _to_02_request(app: dict) -> dict:
    """03's application shape -> 02's `assessment_mode_code=1` (new application) shape."""
    is_joint = app.get("is_joint_application", False)
    req = {
        "decision_id": app["decision_id"], "decision_date": app["decision_date"],
        "product_code": app["product_code"], "channel_code": app["channel_code"],
        "segment_code": app.get("segment_code", 1), "assessment_mode_code": 1,
        "is_joint_application": is_joint, "risk_grade": app["risk_grade"],
        "applicant1_dependants_count": app.get("dependants_count", 0),
        "applicant1_employment_type_code": app.get("employment_type_code", 0),
        "applicant1_payslip_income": app.get("payslip_income") or 0.0,
        "applicant1_variable_pay_history": app.get("variable_pay_history") or [0.0],
        "applicant1_declared_expenses": app.get("declared_expenses") or {},
        "applicant1_statement_expenses": app.get("statement_expenses") or {},
        "applicant1_bureau_accounts": app.get("bureau_accounts") or [dict(_EMPTY_ACCOUNT)],
        "applicant1_internal_accounts": app.get("internal_accounts") or [dict(_EMPTY_ACCOUNT)],
        "applicant2_dependants_count": app.get("applicant2_dependants_count", 0) if is_joint else 0,
        "applicant2_employment_type_code": app.get("applicant2_employment_type_code", 0) if is_joint else 0,
        "applicant2_variable_pay_history": (app.get("applicant2_variable_pay_history") or [0.0]) if is_joint else [0.0],
        "applicant2_declared_expenses": (app.get("applicant2_declared_expenses") or {}) if is_joint else {},
        "applicant2_statement_expenses": (app.get("applicant2_statement_expenses") or {}) if is_joint else {},
        "applicant2_bureau_accounts": (app.get("applicant2_bureau_accounts") or [dict(_EMPTY_ACCOUNT)]) if is_joint
        else [dict(_EMPTY_ACCOUNT)],
        "applicant2_internal_accounts": (app.get("applicant2_internal_accounts") or [dict(_EMPTY_ACCOUNT)]) if is_joint
        else [dict(_EMPTY_ACCOUNT)],
        "bureau_as_of_date": app.get("bureau_as_of_date") or app["decision_date"],
        "court_ordered_deductions": app.get("court_ordered_deductions") or 0.0,
    }
    return req


_AFFORDABILITY_ENGINE = None


def _engine():
    global _AFFORDABILITY_ENGINE
    if _AFFORDABILITY_ENGINE is None:
        from decider import Engine
        _AFFORDABILITY_ENGINE = Engine().bind(affordability_pipeline.build(), mode="interpreted")
    return _AFFORDABILITY_ENGINE


def assess(app: dict) -> dict:
    """Calls 02's pipeline once; returns the fields 03 §5.6 consumes, prefixed
    `affordability_*` where 03 needs its own name for the same concept, and
    passed through unchanged where 03 §5.6 names 02's field directly."""
    req = _to_02_request(app)
    record = dict(req)
    record["decision_date"] = date.fromisoformat(record["decision_date"]) if isinstance(record["decision_date"], str) \
        else record["decision_date"]
    record["bureau_as_of_date"] = date.fromisoformat(record["bureau_as_of_date"]) if \
        isinstance(record["bureau_as_of_date"], str) else record["bureau_as_of_date"]
    for key in ("applicant1_bureau_accounts", "applicant1_internal_accounts",
                "applicant2_bureau_accounts", "applicant2_internal_accounts"):
        for acc in record.get(key, []):
            if isinstance(acc.get("opened_date"), str):
                acc["opened_date"] = date.fromisoformat(acc["opened_date"])
    out = _engine().score(record)
    return {
        "max_affordable_instalment": out["max_affordable_instalment"],
        "discretionary_income": out["discretionary_income"],
        "affordability_verdict_code": out["affordability_verdict_code"],
        "living_expenses": out["living_expenses"],
        "existing_obligations": out["existing_obligations"],
        "affordability_income_verification_tier": out["income_verification_tier"],
        "affordability_norm_table_version": out["norm_table_version"],
        "affordability_adjustment_set_id": out["adjustment_set_id"],
        "affordability_adjustments_applied": out["adjustments_applied"],
        "affordability_evidence_sufficiency_code": out["evidence_sufficiency_code"],
        "affordability_decline_reason_codes": out["decline_reason_codes"],
    }


class Account(TypedDict, total=False):
    opened_date: date


_READS = [
    "decision_id", "decision_date", "product_code", "channel_code", "segment_code", "is_joint_application",
    "risk_grade", "dependants_count", "employment_type_code", "payslip_income", "variable_pay_history",
    "declared_expenses", "statement_expenses", "bureau_accounts", "internal_accounts",
    "applicant2_dependants_count", "applicant2_employment_type_code", "applicant2_variable_pay_history",
    "applicant2_declared_expenses", "applicant2_statement_expenses", "applicant2_bureau_accounts",
    "applicant2_internal_accounts", "bureau_as_of_date", "court_ordered_deductions",
]
_WRITES = [
    "max_affordable_instalment", "discretionary_income", "affordability_verdict_code", "living_expenses",
    "existing_obligations", "affordability_income_verification_tier", "affordability_norm_table_version",
    "affordability_adjustment_set_id", "affordability_adjustments_applied",
    "affordability_evidence_sufficiency_code", "affordability_decline_reason_codes",
]


def _assess_frame(df):
    import polars as pl
    results = [assess(row) for row in df.select(_READS).to_dicts()]
    return df.with_columns(pl.DataFrame(results))


def build_affordability_step():
    """One `frame_step`, not a scalar `step()`: `assess()` needs the whole application
    record (ragged account lists, nested expense structs) to build 02's request shape,
    exactly the row-shaped work `frame_step` exists for (BRIEF)."""
    from decider import frame_step
    # Typed so a served JSON request's account date strings arrive as dates.
    accounts = ("bureau_accounts", "internal_accounts", "applicant2_bureau_accounts", "applicant2_internal_accounts")
    reads = {**dict.fromkeys(_READS, Any), **dict.fromkeys(accounts, list[Account])}
    return frame_step(_assess_frame, reads=reads, writes=_WRITES)
