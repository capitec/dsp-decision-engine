import polars as pl
import pytest

from decider.engine.boundary.nulls import MissingInputError, check_required
from decider.engine.ir.decls import Input, NullPolicy


def test_a_null_in_a_required_input_names_the_input_the_step_path_and_the_row_count():
    frame = pl.DataFrame({"instalment": [100.0] * 7 + [None, 100.0] * 2 + [100.0] * 991})
    with pytest.raises(MissingInputError) as exc_info:
        check_required(frame, [Input("instalment", float)], path="affordability/instalment")
    exc = exc_info.value
    assert (exc.input, exc.path, exc.null_count) == ("instalment", "affordability/instalment", 2)
    msg = str(exc)
    assert "input 'instalment'" in msg and "step 'affordability/instalment'" in msg
    assert "2 null row(s) of 1002" in msg and "missing_as" in msg and "| None" in msg


def test_a_missing_input_error_is_a_value_error():
    with pytest.raises(ValueError, match="application_id"):
        check_required(pl.DataFrame({"application_id": [1, None, 3]}), [Input("application_id", int)])


def test_a_required_column_absent_from_the_frame_raises_by_name():
    with pytest.raises(MissingInputError, match="'missing_col'.*not in the input frame") as exc_info:
        check_required(pl.DataFrame({"a": [1.0, 2.0]}), [Input("missing_col", float)], path="p")
    assert exc_info.value.path == "p"


def test_the_first_offending_required_input_in_declaration_order_is_reported():
    frame = pl.DataFrame({"a": [1.0, 2.0], "b": [None, 1.0], "c": [None, None]}, schema_overrides={"c": pl.Float64})
    with pytest.raises(MissingInputError) as exc_info:
        check_required(frame, [Input("a", float), Input("b", float), Input("c", float)])
    assert exc_info.value.input == "b" and exc_info.value.null_count == 1


def test_a_clean_required_column_passes():
    check_required(pl.DataFrame({"x": [1.0, 2.0, 3.0]}), [Input("x", float)])


def test_only_required_inputs_are_checked_and_a_null_dtype_column_counts_as_all_null():
    frame = pl.DataFrame({"x": [1.0, None], "z": [None, None], "o": [None, 2.0]})
    assert frame.schema["z"] == pl.Null
    check_required(frame, [Input("x", float, NullPolicy.MISSING_AS, fill=0.0),
                           Input("o", float | None, NullPolicy.OPTIONAL)])
    with pytest.raises(MissingInputError, match="'z'.*2 null row"):
        check_required(frame, [Input("z", float)])


def test_a_zero_row_frame_is_not_an_error():
    check_required(pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)}), [Input("x", float)])
