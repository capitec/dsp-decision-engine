from __future__ import annotations

import polars as pl

from decider import flow, missing_as, param
from decider.data import preflight


def _flow():
    def ratio(income: float, debt: float) -> float:
        return debt / income

    def affordable(ratio: float, limit: float = param(0.4)) -> bool:
        return ratio <= limit

    return flow(ratio, affordable, name="afford")


def test_preflight_ok_for_compatible_data():
    frame = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})
    report = preflight(_flow(), frame)
    assert report.ok
    assert report.missing_columns == ()
    assert report.missing_values == ()
    assert report.cost.rows == 2


def test_preflight_reports_missing_required_column():
    frame = pl.DataFrame({"income": [1000.0], "extra": [1.0]})
    report = preflight(_flow(), frame)
    assert not report.ok
    assert report.missing_columns == ("debt",)
    assert any("debt" in e for e in report.errors)


def test_preflight_reports_missing_required_values():
    frame = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, None]})
    report = preflight(_flow(), frame)
    assert not report.ok
    assert report.missing_values == (("debt", 1),)


def test_preflight_accepts_filled_missing_values():
    def f(income: float, debt: float = missing_as(0.0)) -> float:
        return debt / income

    frame = pl.DataFrame({"income": [1000.0], "debt": [None]})
    assert preflight(flow(f, name="g"), frame).ok


def test_preflight_rejects_invalid_params():
    frame = pl.DataFrame({"income": [1000.0], "debt": [200.0]})
    report = preflight(_flow(), frame, params={"afford": {"affordable": {"limit": "high"}}})
    assert not report.ok
    assert any("limit" in e for e in report.errors)


def test_preflight_reports_cost_estimate_and_capabilities():
    frame = pl.DataFrame({"income": [1000.0], "debt": [200.0]})
    report = preflight(_flow(), frame)
    assert report.cost.calls == 2
    assert report.cost.row_operations == 2
    assert report.contract_version
    assert report.modes
