import numpy as np
import polars as pl
import pytest

from decider.engine.boundary.writeback import (
    DtypeGroup,
    KernelOutputs,
    Layout,
    resolve_kept_input_columns,
    row_to_dict,
    to_series,
    write_back,
)


# --- DtypeGroup shape validation ---------------------------------------------

def test_dtype_group_rejects_a_1d_array():
    with pytest.raises(ValueError, match="2D"):
        DtypeGroup(names=("a",), array=np.array([1.0, 2.0]), layout=Layout.COLUMN_MAJOR)


def test_dtype_group_rejects_a_name_count_mismatch_column_major():
    with pytest.raises(ValueError, match="does not match"):
        DtypeGroup(names=("a", "b"), array=np.zeros((1, 3)), layout=Layout.COLUMN_MAJOR)


def test_dtype_group_rejects_a_name_count_mismatch_row_major():
    with pytest.raises(ValueError, match="does not match"):
        DtypeGroup(names=("a", "b"), array=np.zeros((3, 1)), layout=Layout.ROW_MAJOR)


# --- to_series: column-major reaches zero-copy -------------------------------

def test_column_major_to_series_is_zero_copy_per_column():
    # shape (n_cols=2, n_rows=3): row i of the array IS output column i.
    arr = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    group = DtypeGroup(names=("a", "b"), array=arr, layout=Layout.COLUMN_MAJOR)
    series = to_series(group)
    assert [s.name for s in series] == ["a", "b"]
    assert series[0].to_list() == [1.0, 2.0, 3.0]
    assert series[1].to_list() == [10.0, 20.0, 30.0]
    # mutate the backing array; a genuinely zero-copy series reflects it
    arr[0, 0] = 999.0
    assert series[0][0] == 999.0


def test_row_major_to_series_is_correct_even_though_strided():
    # shape (n_rows=3, n_cols=2): column i is a STRIDED slice of the array.
    arr = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    group = DtypeGroup(names=("a", "b"), array=arr, layout=Layout.ROW_MAJOR)
    series = to_series(group)
    assert series[0].to_list() == [1.0, 2.0, 3.0]
    assert series[1].to_list() == [10.0, 20.0, 30.0]


# --- write_back: additive assembly via hstack --------------------------------

def test_write_back_appends_computed_columns_to_every_input_column():
    frame = pl.DataFrame({"net_income": [100.0, 200.0], "expenses": [10.0, 20.0]})
    arr = np.array([[90.0, 180.0]])  # one output column: disposable_income
    outputs = KernelOutputs(float64=DtypeGroup(("disposable_income",), arr, Layout.COLUMN_MAJOR))
    out = write_back(frame, outputs)
    assert out.columns == ["net_income", "expenses", "disposable_income"]
    assert out["disposable_income"].to_list() == [90.0, 180.0]


def test_write_back_with_keep_drops_a_declared_column():
    frame = pl.DataFrame({"id_number": [1, 2], "net_income": [100.0, 200.0]})
    outputs = KernelOutputs()  # no computed columns for this test
    kept = resolve_kept_input_columns(frame.columns, dropped=("id_number",))
    out = write_back(frame, outputs, keep=kept)
    assert out.columns == ["net_income"]


def test_write_back_with_no_computed_outputs_returns_the_kept_frame_unchanged():
    frame = pl.DataFrame({"x": [1.0, 2.0]})
    out = write_back(frame, KernelOutputs())
    assert out.equals(frame)


def test_write_back_combines_all_three_dtype_groups():
    frame = pl.DataFrame({"id": [1, 2]})
    outputs = KernelOutputs(
        float64=DtypeGroup(("score",), np.array([[0.5, 0.9]]), Layout.COLUMN_MAJOR),
        int64=DtypeGroup(("term",), np.array([[36, 48]]), Layout.COLUMN_MAJOR),
        bool_=DtypeGroup(("approved",), np.array([[True, False]]), Layout.COLUMN_MAJOR),
    )
    out = write_back(frame, outputs)
    assert out.columns == ["id", "score", "term", "approved"]
    assert out["approved"].to_list() == [True, False]


# --- resolve_kept_input_columns: the frame-column half of "additive" --------

def test_resolve_kept_input_columns_excludes_dropped_and_overwritten():
    kept = resolve_kept_input_columns(
        ["id", "term_cap", "min_net_salary"],
        overwritten=("term_cap",),   # a kernel output rewrites this name
        dropped=("min_net_salary",),
    )
    assert kept == ("id",)


def test_resolve_kept_input_columns_keeps_everything_by_default():
    assert resolve_kept_input_columns(["a", "b"]) == ("a", "b")


# --- row_to_dict: the single-record path -------------------------------------

def test_row_to_dict_merges_all_groups_for_one_record():
    outputs = KernelOutputs(
        float64=DtypeGroup(("cap_by_income_band",), np.array([[48.0]]), Layout.ROW_MAJOR),
        bool_=DtypeGroup(("approved",), np.array([[True]]), Layout.ROW_MAJOR),
    )
    result = row_to_dict(outputs)
    assert result == {"cap_by_income_band": 48.0, "approved": True}
    assert isinstance(result["approved"], bool)


def test_row_to_dict_rejects_more_than_one_row():
    outputs = KernelOutputs(float64=DtypeGroup(("x",), np.array([[1.0], [2.0]]), Layout.ROW_MAJOR))
    with pytest.raises(ValueError, match="exactly one row"):
        row_to_dict(outputs)


def test_row_to_dict_rejects_a_column_major_group():
    outputs = KernelOutputs(float64=DtypeGroup(("x",), np.array([[1.0, 2.0]]), Layout.COLUMN_MAJOR))
    with pytest.raises(ValueError, match="row-major"):
        row_to_dict(outputs)
