"""outputs: which rows and columns change between two versions over a frame."""
import polars as pl
import pytest

from decider import flow
from decider.lab import diff
from decider.lab.diff import _outputs


def _version(rate: float, label: str = "a", offer_cents: int = 100):
    def score(x: float) -> float:
        return x * rate

    def tier(x: float) -> str:
        return label if x < 2.5 else "z"

    def offer(x: float) -> int:
        return offer_cents * int(x)

    return flow(score, tier, offer, name="t")


FRAME = pl.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})


def test_identical_versions_change_nothing():
    o = diff.outputs(_version(2.0), _version(2.0), FRAME, key="id")
    assert (o.rows_compared, o.rows_changed, o.columns) == (3, 0, {})
    assert o.table().is_empty()


def test_a_changed_column_reports_size_and_the_rows_by_key():
    o = diff.outputs(_version(2.0), _version(3.0), FRAME, key="id")
    c = o.columns["score"]
    assert c.changed == 3 and c.max_abs == pytest.approx(3.0) and c.mean_delta == pytest.approx(2.0)
    assert c.max_rel == pytest.approx(0.5)
    assert c.samples[0] == {"id": "a", "old": 2.0, "new": 3.0}
    assert o.rows_changed == 3
    assert set(o.table().columns) == {"id", "column", "old", "new", "delta"}


def test_without_a_key_rows_are_labelled_by_position():
    assert diff.outputs(_version(2.0), _version(3.0), FRAME).columns["score"].samples[0]["row"] == 0


def test_a_float_difference_inside_the_tolerance_is_not_a_change():
    assert diff.outputs(_version(2.0), _version(2.0 + 1e-12), FRAME).rows_changed == 0
    assert diff.outputs(_version(2.0), _version(2.0 + 1e-12), FRAME, rtol=0, atol=0).rows_changed == 3


def test_integer_deltas_are_exact():
    o = diff.outputs(_version(2.0), _version(2.0, offer_cents=101), FRAME)
    c = o.columns["offer"]
    assert isinstance(c.max_abs, int) and c.max_abs == 3


def test_a_non_numeric_change_has_no_delta():
    o = diff.outputs(_version(2.0), _version(2.0, label="b"), FRAME)
    c = o.columns["tier"]
    assert c.changed == 2 and c.max_abs is None


def test_nan_equals_nan_and_null_equals_null_but_not_a_value():
    old = pl.Series([1.0, float("nan"), None, 4.0])
    new = pl.Series([1.0, float("nan"), None, None])
    assert _outputs.changed(old, new).to_list() == [False, False, False, True]
    assert _outputs.changed(pl.Series([1.0]), pl.Series([float("nan")])).to_list() == [True]


def test_columns_present_in_one_version_only_are_listed():
    def extra(x: float) -> float:
        return x

    old, new = flow(extra, name="t"), _version(2.0)
    o = diff.outputs(old, new, FRAME)
    assert o.added_columns == ["score", "tier", "offer"] and o.removed_columns == ["extra"]
    assert diff.outputs(new, old, FRAME).removed_columns == o.added_columns


def test_a_dtype_change_is_reported():
    o = _outputs.compare_frames(pl.DataFrame({"v": [1, 2]}), pl.DataFrame({"v": [1.0, 3.0]}))
    assert o.columns["v"].dtype_change == ("Int64", "Float64") and o.columns["v"].changed == 1


def test_an_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="key"):
        diff.outputs(_version(2.0), _version(3.0), FRAME, key="nope")
