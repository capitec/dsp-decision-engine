from __future__ import annotations

import polars as pl

from decider import flow, frame_step, param
from decider.contract import RecordRef, describe
from decider.data import ExecutionScope, check_scope, frame_steps


def _scalar_flow():
    def ratio(income: float, debt: float) -> float:
        return debt / income

    def affordable(ratio: float, limit: float = param(0.4)) -> bool:
        return ratio <= limit

    return flow(ratio, affordable, name="afford")


def _frame_flow():
    @frame_step(reads=["amount"], writes=["total"])
    def total(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("amount").sum().alias("total"))

    def scaled(amount: float, total: float) -> float:
        return amount / total

    return flow(total, scaled, name="normalise")


def test_frame_steps_detects_frame_kind_nodes():
    assert [s.path for s in frame_steps(describe(_frame_flow()))] == ["normalise/total"]
    assert frame_steps(describe(_scalar_flow())) == ()


def test_selected_record_is_valid_without_frame_steps():
    check = check_scope(describe(_scalar_flow()), ExecutionScope.SELECTED_RECORD)
    assert check.effective is ExecutionScope.SELECTED_RECORD
    assert not check.redirected


def test_record_only_scope_redirects_to_whole_frame_over_frame_steps():
    check = check_scope(describe(_frame_flow()), ExecutionScope.SELECTED_RECORD,
                        focus=RecordRef(dataset="loans.parquet", key={"client_id": "C-1"}))
    assert check.redirected
    assert check.effective is ExecutionScope.WHOLE_FRAME
    assert check.frame_steps == ("normalise/total",)
    assert check.hybrid


def test_whole_frame_is_always_valid():
    check = check_scope(describe(_frame_flow()), ExecutionScope.WHOLE_FRAME)
    assert check.effective is ExecutionScope.WHOLE_FRAME
    assert not check.redirected
    assert not check.hybrid
