"""Each null policy crossed with each kind, through the one path a column takes."""
import numpy as np
import polars as pl
import pytest

pytest.importorskip("decider.engine.boundary._arrow._shim", exc_type=ImportError, reason="the Arrow shim can't be built here")

from decider.engine.boundary.extract import extract_frame  # noqa: E402
from decider.engine.boundary.nulls import MissingInputError  # noqa: E402
from decider.engine.ir.decls import FeatureKind as K  # noqa: E402
from decider.engine.ir.decls import Input, NullPolicy  # noqa: E402

# (annotation, a column with a null at row 1, the fill to declare, the value the kernel dtype gives the fill)
_KIND_CASES = {
    K.F64: (float, pl.Series([1.5, None, 3.0]), -1.0, -1.0),
    K.I64: (int, pl.Series([2 ** 62 + 1, None, 3]), -7, -7),
    K.BOOL: (bool, pl.Series([False, None, True]), True, True),
    K.CODE: (str, pl.Series(["a", None, "b"]), -1, -1),
}


@pytest.mark.parametrize("kind", list(_KIND_CASES))
def test_missing_as_fills_the_null_with_the_declared_value(kind):
    annotation, series, fill, filled = _KIND_CASES[kind]
    col = extract_frame(pl.DataFrame({"x": series}),
                        [Input("x", annotation, NullPolicy.MISSING_AS, fill=fill)]).columns["x"]
    assert col.values[1] == filled and col.plan.kind is kind and col.validity is None


@pytest.mark.parametrize("kind", list(_KIND_CASES))
def test_optional_masks_and_never_fills(kind):
    annotation, series, _, _ = _KIND_CASES[kind]
    col = extract_frame(pl.DataFrame({"x": series}), [Input("x", annotation, NullPolicy.OPTIONAL)]).columns["x"]
    assert col.validity.tolist() == [True, False, True]
    if kind is not K.CODE:
        assert col.values[0] == series[0]


@pytest.mark.parametrize("kind", list(_KIND_CASES))
def test_required_fails_on_a_null_for_every_kind(kind):
    annotation, series, _, _ = _KIND_CASES[kind]
    with pytest.raises(MissingInputError, match="'x'.*1 null row"):
        extract_frame(pl.DataFrame({"x": series}), [Input("x", annotation, NullPolicy.REQUIRED)])


def test_a_bytes_input_has_no_fill_and_reports_null_spans():
    frame = pl.DataFrame({"s": ["ab", None]})
    col = extract_frame(frame, [Input("s", bytes, NullPolicy.OPTIONAL)]).columns["s"]
    assert col.validity.tolist() == [True, False] and col.values[1].tolist() == [0, -1]
    with pytest.raises(ValueError, match="STR feature has no fill"):
        extract_frame(frame, [Input("s", bytes, NullPolicy.MISSING_AS, fill=0)])


def test_an_absent_column_and_an_all_null_column_produce_the_same_column_per_policy():
    present = pl.DataFrame({"k": [1, 2, 3], "x": pl.Series([None, None, None], dtype=pl.Float64)})
    absent = pl.DataFrame({"k": [1, 2, 3]})
    for policy, fill in ((NullPolicy.OPTIONAL, None), (NullPolicy.MISSING_AS, 7.0)):
        decl = Input("x", float | None if fill is None else float, policy, fill=fill)
        a = extract_frame(present, [Input("k", int), decl]).columns["x"]
        b = extract_frame(absent, [Input("k", int), decl]).columns["x"]
        assert a.values.dtype == b.values.dtype == np.float64
        if policy is NullPolicy.OPTIONAL:
            assert a.validity.tolist() == b.validity.tolist() == [False] * 3
        else:
            assert a.values.tolist() == b.values.tolist() == [7.0] * 3


def test_every_policy_is_correct_at_once_in_one_frame():
    frame = pl.DataFrame({
        "id": [1, 2, 3],
        "score": [600.0, None, 700.0],
        "income": [5000.0, None, 7000.0],
    })
    inputs = [
        Input("id", int),
        Input("score", float, NullPolicy.MISSING_AS, fill=-1.0),
        Input("income", float | None, NullPolicy.OPTIONAL),
    ]
    result = extract_frame(frame, inputs)
    assert result.columns["score"].values.tolist() == [600.0, -1.0, 700.0]
    assert result.columns["income"].validity.tolist() == [True, False, True]
    assert result.columns["id"].values.tolist() == [1, 2, 3] and result.columns["id"].values.dtype == np.int64
