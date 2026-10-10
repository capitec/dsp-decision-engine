from __future__ import annotations

import polars as pl

from decikit import flow, missing_as, param
from decikit.lab import checks

MISSING = ["missing_required"]
UNKNOWN = ["invalid_params_doc"]


def _flow():
    def ratio(income: float, debt: float) -> float:
        return debt / income

    def affordable(ratio: float, limit: float = param(0.4)) -> bool:
        return ratio <= limit

    return flow(ratio, affordable, name="afford")


def test_missing_required_is_quiet_for_compatible_data():
    frame = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})
    assert checks.run(_flow(), MISSING, data=frame).findings == ()


def test_missing_required_column_is_reported_at_its_reader():
    report = checks.run(_flow(), MISSING, data=pl.DataFrame({"income": [1000.0], "extra": [1.0]}))
    (f,) = report.findings
    assert (f.code, f.detail["column"], f.step.path) == ("DCD504", "debt", "afford/ratio")
    assert not report.ok


def test_null_in_required_column_is_reported_with_its_count():
    report = checks.run(_flow(), MISSING, data=pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, None]}))
    (f,) = report.findings
    assert f.detail == {"column": "debt", "nulls": 1}


def test_filled_missing_values_are_quiet():
    def f(income: float, debt: float = missing_as(0.0)) -> float:
        return debt / income

    frame = pl.DataFrame({"income": [1000.0], "debt": [None]})
    assert checks.run(flow(f, name="g"), MISSING, data=frame).findings == ()


def test_missing_required_is_skipped_without_data():
    assert [s.check for s in checks.run(_flow(), MISSING).skipped] == ["missing_required"]


def test_invalid_params_doc_is_reported():
    report = checks.run(_flow(), UNKNOWN, params={"afford": {"affordble": {"limit": 0.5}}})
    (f,) = report.findings
    assert f.code == "DCD304"
    assert "affordble" in f.message


def test_params_doc_invalid_value_is_reported_at_its_step():
    report = checks.run(_flow(), UNKNOWN, params={"afford": {"affordable": {"limit": "high"}}})
    (f,) = report.findings
    assert f.step.path == "afford/affordable"
    assert "limit" in f.message


def test_params_doc_known_keys_are_quiet_and_no_document_is_skipped_quietly():
    assert checks.run(_flow(), UNKNOWN, params={"afford": {"affordable": {"limit": 0.5}}}).findings == ()
    assert checks.run(_flow(), UNKNOWN).findings == ()


def test_both_are_registered_but_not_standard():
    rows = {c["alias"]: c for c in checks.catalogue()}
    assert rows["missing_required"]["needs_data"] and rows["missing_required"]["code"] == "DCD504"
    assert rows["invalid_params_doc"]["code"] == "DCD304"
    assert not {"missing_required", "invalid_params_doc"} & set(checks.standard)
