import ctypes
import threading
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

pytest.importorskip("decider.engine.boundary._arrow._shim", exc_type=ImportError, reason="the Arrow shim can't be built here")

from decider.engine.boundary._arrow.plan import ArrowKindError  # noqa: E402
from decider.engine.boundary.dtypes import NeedsKernelSplit  # noqa: E402
from decider.engine.boundary.extract import _local, _schema_plan, extract_frame  # noqa: E402
from decider.engine.boundary.nulls import MissingInputError  # noqa: E402
from decider.engine.ir.decls import Input, NullPolicy  # noqa: E402


def _spans(values: np.ndarray) -> list:
    return [None if ln < 0 else ctypes.string_at(int(a), int(ln)) for a, ln in values.tolist()]


def _nullable(values, valid):
    return [v if ok else None for v, ok in zip(values, valid)]


# --- every dtype round-trips ---------------------------------------------------

_DTYPE_ROWS = {
    "f64": (pl.Series([1.5, None, -0.0], dtype=pl.Float64), float),
    "f32": (pl.Series([2.5, None, 1e-3], dtype=pl.Float32), float),
    "i8": (pl.Series([-128, None, 127], dtype=pl.Int8), int),
    "i16": (pl.Series([-32768, None, 32767], dtype=pl.Int16), int),
    "i32": (pl.Series([-(2 ** 31), None, 2 ** 31 - 1], dtype=pl.Int32), int),
    "i64": (pl.Series([2 ** 62 + 1, None, -(2 ** 63)], dtype=pl.Int64), int),   # above 2**53: no float round trip
    "u8": (pl.Series([255, None, 0], dtype=pl.UInt8), int),
    "u16": (pl.Series([65535, None, 1], dtype=pl.UInt16), int),
    "u32": (pl.Series([2 ** 32 - 1, None, 2], dtype=pl.UInt32), int),
    "u64": (pl.Series([2 ** 62 + 1, None, 2 ** 63 - 1], dtype=pl.UInt64), int),
    "bool": (pl.Series([True, None, False], dtype=pl.Boolean), bool),
    "date": (pl.Series([date(2024, 2, 29), None, date(1969, 12, 31)], dtype=pl.Date), int),
    "dt": (pl.Series([datetime(2024, 2, 29, 12, 34, 56, 789), None, datetime(1970, 1, 1)], dtype=pl.Datetime("us")), int),
    "dt_ms": (pl.Series([datetime(2024, 2, 29, 12, 34, 56), None, datetime(1970, 1, 1)], dtype=pl.Datetime("ms")), int),
    "dur": (pl.Series([timedelta(days=1, seconds=5), None, timedelta(0)], dtype=pl.Duration("ms")), int),
    "time": (pl.Series([time(12, 34, 56), None, time(0)], dtype=pl.Time), int),
}


@pytest.mark.parametrize("name", list(_DTYPE_ROWS))
def test_every_dtype_of_the_table_round_trips_exactly_as_python_sees_it(name):
    series, annotation = _DTYPE_ROWS[name]
    frame = pl.DataFrame({name: series})
    out = extract_frame(frame, [Input(name, annotation, NullPolicy.OPTIONAL)]).columns[name]
    expected_dtype = {float: np.float64, int: np.int64, bool: np.bool_}[annotation]
    assert out.values.dtype == expected_dtype
    expected = series.to_physical().to_list() if series.dtype.is_temporal() else series.to_list()
    got = _nullable(out.values.tolist(), out.validity)
    if annotation is float:
        assert [None if v is None else float(v) for v in expected] == got
    else:
        assert expected == got
    assert out.validity.tolist() == [True, False, True]


def test_an_int64_above_2_53_survives_exactly_in_int64_and_uint64():
    frame = pl.DataFrame({
        "i": pl.Series([2 ** 62 + 1], dtype=pl.Int64), "u": pl.Series([2 ** 62 + 1], dtype=pl.UInt64),
    })
    ex = extract_frame(frame, [Input("i", int), Input("u", int)])
    assert ex.columns["i"].values.tolist() == [2 ** 62 + 1] and ex.columns["u"].values.tolist() == [2 ** 62 + 1]
    assert float(2 ** 62 + 1) != 2 ** 62 + 1                          # the value a float64 path would have produced


def test_floats_stay_floats_ints_stay_ints_bools_stay_bools():
    frame = pl.DataFrame({"f": [1.5], "i": [7], "b": [True], "s": ["k"], "c": pl.Series(["k"], dtype=pl.Categorical)})
    ex = extract_frame(frame, [Input("f", float), Input("i", int), Input("b", bool), Input("s", str), Input("c", str)])
    assert [ex.columns[n].values.dtype for n in ("f", "i", "b", "s", "c")] == [np.float64, np.int64, np.bool_, np.int32, np.int32]
    for ec in ex.columns.values():
        assert ec.values.flags.writeable is False and ec.values.flags.c_contiguous


def test_a_string_column_declared_bytes_arrives_as_spans_into_polars_memory():
    values = ["", "twelve chars", "thirteen char", None, "héllo wörld", "日本語のテキスト", "a" * 100]
    frame = pl.DataFrame({"s": values})
    ex = extract_frame(frame, [Input("s", bytes, NullPolicy.OPTIONAL)])
    col = ex.columns["s"]
    assert col.values.shape == (7, 2) and col.values.dtype == np.int64
    assert _spans(col.values) == [None if v is None else v.encode() for v in values]
    assert col.validity.tolist() == [v is not None for v in values]
    assert col.values[3].tolist() == [0, -1]                                                        # the null span
    del frame
    assert _spans(col.values)[6] == b"a" * 100                                                  # kernel_frame keeps the bytes alive


def test_a_str_input_arrives_as_dictionary_codes_with_the_exported_dictionary():
    # The categories are the exported dictionary, not `cat.get_categories()`:
    # polars' physical codes are process-global, its Arrow dictionary is per batch.
    other = pl.Series(["zzz", "public"]).cast(pl.Categorical)            # intern something first
    assert other.to_physical().to_list() is not None
    frame = pl.DataFrame({"s": ["public", "private", None, "public", "state"]})
    ex = extract_frame(frame, [Input("s", str, NullPolicy.MISSING_AS, fill=-1)])
    col = ex.columns["s"]
    cats = col.categories
    assert [None if c < 0 else cats[c] for c in col.values.tolist()] == ["public", "private", None, "public", "state"]
    assert len(cats) == 3
    assert frame.schema["s"] == pl.String                                # the caller's column was not re-typed


def test_categorical_and_enum_inputs_read_their_dictionary_indices_and_dictionaries():
    frame = pl.DataFrame({
        "c": pl.Series(["x", "y", None, "x"], dtype=pl.Categorical),
        "e": pl.Series(["lo", "hi", "lo", None], dtype=pl.Enum(["lo", "mid", "hi"])),
    })
    ex = extract_frame(frame, [Input("c", str, NullPolicy.OPTIONAL), Input("e", str, NullPolicy.OPTIONAL)])
    c, e = ex.columns["c"], ex.columns["e"]
    assert [None if not ok else c.categories[v] for v, ok in zip(c.values.tolist(), c.validity)] == ["x", "y", None, "x"]
    assert e.categories == ("lo", "mid", "hi") and e.values.tolist()[:3] == [0, 2, 0] and not e.validity[3]


# --- nulls per kind and per policy --------------------------------------------

def test_missing_as_fills_nulls_in_the_gather_and_needs_no_mask():
    frame = pl.DataFrame({"f": [10.0, None, 30.0], "i": [None, 2, None], "b": [None, True, None]})
    ex = extract_frame(frame, [
        Input("f", float, NullPolicy.MISSING_AS, fill=0.0),
        Input("i", int, NullPolicy.MISSING_AS, fill=-7),
        Input("b", bool, NullPolicy.MISSING_AS, fill=True),
    ])
    f, i, b = ex.columns["f"], ex.columns["i"], ex.columns["b"]
    assert f.values.tolist() == [10.0, 0.0, 30.0]
    assert i.values.tolist() == [-7, 2, -7]
    assert b.values.tolist() == [True, True, True]
    assert f.validity is None and i.validity is None and b.validity is None


def test_a_clean_missing_as_column_passes_through_unchanged():
    ex = extract_frame(pl.DataFrame({"x": [1.0, 2.0]}), [Input("x", float, NullPolicy.MISSING_AS, fill=-1.0)])
    assert ex.columns["x"].values.tolist() == [1.0, 2.0]


def test_optional_keeps_a_validity_mask_and_never_fills():
    ex = extract_frame(pl.DataFrame({"x": [10.0, None]}), [Input("x", float | None, NullPolicy.OPTIONAL)])
    col = ex.columns["x"]
    assert col.validity.tolist() == [True, False]
    assert np.isnan(col.values[1])                                       # the default under a null, never read by a kernel


def test_optional_clean_column_still_gets_an_all_true_mask():
    ex = extract_frame(pl.DataFrame({"x": [1.0, 2.0]}), [Input("x", float | None, NullPolicy.OPTIONAL)])
    assert ex.columns["x"].validity.tolist() == [True, True]


def test_the_garbage_slot_under_a_null_is_never_read():
    # The value slot under a null holds leftover data, not zero.
    picked = pl.Series("m", [True, False, True])
    source = pl.Series("v", [9.0, 9.0, 9.0])
    s = pl.DataFrame({"m": picked, "v": source}).select(
        pl.when(pl.col("m")).then(pl.col("v")).otherwise(None).alias("v")
    )["v"]
    assert s.null_count() == 1
    assert s._get_buffers()["values"].to_numpy(allow_copy=False)[1] == 9.0  # the garbage, confirmed present
    ex = extract_frame(pl.DataFrame({"v": s}), [Input("v", float, NullPolicy.MISSING_AS, fill=0.0)])
    assert ex.columns["v"].values.tolist() == [9.0, 0.0, 9.0]


def test_a_left_join_null_is_filled_too():
    left = pl.DataFrame({"k": [1, 2, 3]})
    right = pl.DataFrame({"k": [1, 3], "v": [9.0, 9.0]})
    frame = left.join(right, on="k", how="left")
    ex = extract_frame(frame, [Input("k", int), Input("v", float, NullPolicy.MISSING_AS, fill=-1.0)])
    assert ex.columns["v"].values.tolist() == [9.0, -1.0, 9.0]


def test_an_all_null_null_dtype_column_fills_or_masks_like_any_other():
    frame = pl.DataFrame({"z": [None, None], "k": [1, 2]})
    assert frame.schema["z"] == pl.Null
    ex = extract_frame(frame, [Input("z", float, NullPolicy.MISSING_AS, fill=3.0), Input("k", int)])
    assert ex.columns["z"].values.tolist() == [3.0, 3.0] and ex.columns["z"].plan.cast == pl.Float64()
    ex = extract_frame(frame, [Input("z", int | None, NullPolicy.OPTIONAL), Input("k", int)])
    assert ex.columns["z"].validity.tolist() == [False, False]


# --- frame shapes: fresh, sliced, multi-chunk, null at the ends --------------

def _shape_inputs():
    return [
        Input("f", float), Input("i", int), Input("b", bool), Input("s", str, NullPolicy.MISSING_AS, fill=-1),
        Input("o", float | None, NullPolicy.OPTIONAL), Input("m", int, NullPolicy.MISSING_AS, fill=-9),
        Input("r", bytes, NullPolicy.OPTIONAL),
    ]


def _shape_frame(n=12, seed=3):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "f": rng.uniform(-1e6, 1e6, n),
        "i": rng.integers(-(2 ** 62), 2 ** 62, n, dtype=np.int64),
        "b": rng.integers(0, 2, n).astype(bool),
        "s": rng.choice(["private", "public", "government"], n),
        "o": [None if k % 4 == 1 else float(k) for k in range(n)],
        "m": [None if k in (0, n - 1) else k * 1_000_000_007 for k in range(n)],   # the null on row 0 AND the last row
        "r": [None if k % 5 == 2 else f"row {k} " + "x" * (k % 15) for k in range(n)],
    })


def _canonical(ex, n):
    """Every extracted column as plain Python values, spans decoded, codes mapped to text."""
    out = {}
    for name, ec in ex.columns.items():
        if ec.plan.kind.name == "STR":
            out[name] = _spans(ec.values)
        elif ec.categories is not None:
            out[name] = [None if c < 0 else ec.categories[c] for c in ec.values.tolist()]
        else:
            out[name] = ec.values.tolist()
        if ec.validity is not None:
            # a masked slot's value is never read (NaN under an OPTIONAL null); compare it as None
            out[name] = _nullable(out[name], ec.validity)
            out[f"__valid__{name}"] = ec.validity.tolist()
        assert len(ec.values) == n
    return out


def test_fresh_sliced_and_multi_chunk_frames_extract_identically():
    fresh = _shape_frame()
    reference = _canonical(extract_frame(fresh, _shape_inputs()), 12)

    sliced = _shape_frame()[3:10]                                       # `df[a:b]` carries its Arrow offset on each child
    got = _canonical(extract_frame(sliced, _shape_inputs()), 7)
    assert got == {k: v[3:10] for k, v in reference.items()}

    parts = [_shape_frame()[0:5], _shape_frame()[5:9], _shape_frame()[9:12]]
    multi = pl.concat(parts)
    assert multi.n_chunks() == 3
    got = _canonical(extract_frame(multi, _shape_inputs()), 12)
    assert got == reference
    # `s: str` needs a frame-tier cast, so the export ran on a clone and the
    # caller's frame kept its chunks; without a cast polars rechunks the
    # caller's frame in place. Both halves pinned, so a polars change is noticed.
    assert multi.n_chunks("all") == [3] * multi.width
    no_cast = [decl for decl in _shape_inputs() if decl.name != "s"]
    got = _canonical(extract_frame(multi, no_cast), 12)
    assert got == {k: v for k, v in reference.items() if k != "s"}
    assert multi.n_chunks("all") == [1] * multi.width


@pytest.mark.parametrize("a,b", [(0, 1), (0, 12), (11, 12), (1, 11), (6, 8)])
def test_the_null_on_row_zero_and_on_the_last_row_survives_every_slice(a, b):
    reference = _canonical(extract_frame(_shape_frame(), _shape_inputs()), 12)
    got = _canonical(extract_frame(_shape_frame()[a:b], _shape_inputs()), b - a)
    assert got == {k: v[a:b] for k, v in reference.items()}


def test_a_zero_row_frame_extracts_zero_length_columns_of_the_right_dtype():
    ex = extract_frame(_shape_frame().clear(), _shape_inputs())
    assert ex.kernel_frame.height == 0
    assert {n: (ec.values.shape, ec.values.dtype.name) for n, ec in ex.columns.items()} == {
        "f": ((0,), "float64"), "i": ((0,), "int64"), "b": ((0,), "bool"), "s": ((0,), "int32"),
        "o": ((0,), "float64"), "m": ((0,), "int64"), "r": ((0, 2), "int64"),
    }


# --- required nulls, absent columns ------------------------------------------

def test_a_null_in_a_required_input_fails_before_any_export_naming_input_path_and_count():
    frame = pl.DataFrame({"instalment": [100.0, None, 300.0], "id": [1, 2, 3]})
    with pytest.raises(MissingInputError, match=r"'instalment' of step 'loan/instalment'.*1 null row\(s\) of 3"):
        extract_frame(frame, [Input("instalment", float), Input("id", int)], path="loan/instalment")


def test_a_required_column_entirely_absent_fails_by_name():
    with pytest.raises(MissingInputError, match="'missing_col'.*not in the input frame"):
        extract_frame(pl.DataFrame({"a": [1.0]}), [Input("a", float), Input("missing_col", float)])


def test_absent_columns_are_filled_or_masked_in_the_kinds_own_dtype():
    frame = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    ex = extract_frame(frame, [
        Input("a", float),
        Input("f", float, NullPolicy.MISSING_AS, fill=7.0),
        Input("i", int, NullPolicy.MISSING_AS, fill=-1),
        Input("o", int | None, NullPolicy.OPTIONAL),
        Input("s", str, NullPolicy.OPTIONAL),
        Input("r", bytes, NullPolicy.OPTIONAL),
    ])
    assert ex.columns["f"].values.tolist() == [7.0] * 3 and ex.columns["f"].validity is None
    assert ex.columns["i"].values.tolist() == [-1] * 3 and ex.columns["i"].values.dtype == np.int64
    assert ex.columns["o"].validity.tolist() == [False] * 3 and ex.columns["o"].values.dtype == np.float64
    assert ex.columns["s"].values.dtype == np.int32 and ex.columns["s"].validity.tolist() == [False] * 3
    assert ex.columns["r"].values.shape == (3, 2) and _spans(ex.columns["r"].values) == [None] * 3
    assert frame.columns == ["a"]                                        # the caller's frame is not extended


def test_a_str_column_with_a_text_fill_is_refused_by_name():
    with pytest.raises(ValueError, match="'s' is a `str` column with a string fill"):
        extract_frame(pl.DataFrame({"s": ["a"]}), [Input("s", str, NullPolicy.MISSING_AS, fill="unknown")])


# --- what is refused, and how it says so ---------------------------------------

def test_a_list_or_struct_column_is_reported_for_the_kernel_split_before_any_export():
    for frame, name in ((pl.DataFrame({"amounts": [[1.0, 2.0], [3.0]]}), "amounts"),
                        (pl.DataFrame({"st": [{"a": 1}, {"a": 2}]}), "st")):
        with pytest.raises(NeedsKernelSplit) as exc_info:
            extract_frame(frame, [Input(name, float)])
        assert exc_info.value.column == name and isinstance(exc_info.value, ArrowKindError)


def test_a_pair_the_table_has_no_cast_for_is_refused_by_column_arrow_type_and_kind():
    with pytest.raises(ArrowKindError, match=r"column 's' is Arrow string_view but is declared F64") as exc_info:
        extract_frame(pl.DataFrame({"s": ["a"]}), [Input("s", float)])
    assert (exc_info.value.column, exc_info.value.arrow_type) == ("s", "string_view")
    with pytest.raises(ArrowKindError, match=r"column 'c' is Arrow dictionary\(uint32\)<string_view> but is declared I64"):
        extract_frame(pl.DataFrame({"c": pl.Series(["a"], dtype=pl.Categorical)}), [Input("c", int)])


def test_a_decimal_column_crosses_as_scaled_int64_cents_as_before():
    frame = pl.DataFrame({"amount": pl.Series([Decimal("19.99"), Decimal("5.00")], dtype=pl.Decimal(18, 2))})
    col = extract_frame(frame, [Input("amount", int)]).columns["amount"]
    assert col.values.tolist() == [1999, 500] and col.values.dtype == np.int64 and col.plan.cast == pl.Int64()
    col = extract_frame(frame, [Input("amount", float)]).columns["amount"]
    assert col.values.tolist() == [1999.0, 500.0]


def test_a_decimal_that_overflows_int64_fails_by_name_not_by_crash():
    frame = pl.DataFrame({"m": pl.Series([Decimal("9" * 30)], dtype=pl.Decimal(38, 0))})
    with pytest.raises(NeedsKernelSplit, match=r"column 'm'.*Decimal.*cast to Int64 failed"):
        extract_frame(frame, [Input("m", int)])


# --- the schema plan is decided once; the FrameView is pooled per thread -------

def test_the_schema_plan_is_decided_once_per_inputs_and_frame_schema():
    inputs = (Input("a", float), Input("b", int))
    frame = pl.DataFrame({"a": [1.0], "b": [2]})
    p1 = _schema_plan(inputs, tuple(frame.columns), tuple(frame.dtypes))
    p2 = _schema_plan(inputs, tuple(frame.columns), tuple(frame.dtypes))
    assert p1 is p2
    p3 = _schema_plan(inputs, ("b", "a"), (pl.Int64(), pl.Float64()))
    assert p3 is not p1                                                  # a different column order is a different plan


def test_frame_views_are_pooled_per_thread_not_per_process():
    inputs = [Input("a", float)]
    frame = pl.DataFrame({"a": [1.0, 2.0]})
    extract_frame(frame, inputs)
    plan = _schema_plan(tuple(inputs), tuple(frame.columns), tuple(frame.dtypes)).frame_plan
    mine = _local.views[plan]
    seen = {}

    def worker():
        extract_frame(frame, inputs)
        seen["view"] = _local.views[plan]

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["view"] is not mine and not seen["view"].bound and not mine.bound
    assert _local.views[plan] is mine                                    # this thread's pool is untouched
