import datetime as dt
import json

import polars as pl
import pytest
from pydantic import ValidationError

from decikit._serde import FrameDoc, PolarsSchema, PrimitiveSchema, convert_schema


def _through_json(value):
    return json.loads(json.dumps(value))


def test_a_polars_schema_round_trips_through_json():
    schema = pl.Schema({
        "id": pl.Int64, "at": pl.Datetime("ms", "UTC"), "amount": pl.Decimal(10, 2), "tags": pl.List(pl.String),
        "grid": pl.Array(pl.Int32, (4, 2)), "band": pl.Enum(["low", "high"]), "took": pl.Duration("ms"),
        "kind": pl.Categorical(), "day": pl.Date, "point": pl.Struct({"x": pl.Int8, "ys": pl.List(pl.Float64)}),
    })
    doc = _through_json(PolarsSchema.from_polars(schema).model_dump())
    assert PolarsSchema.model_validate(doc).schema == schema


def test_a_schema_is_written_in_column_order_with_arguments_only_where_needed():
    schema = pl.Schema({"b": pl.Float64, "a": pl.List(pl.String)})
    assert convert_schema(schema) == [("b", "Float64"), ("a", {"type": "List", "inner": "String"})]


def test_a_frame_round_trips_through_json_with_its_dtypes():
    frame = pl.DataFrame({
        "n": [1, 2], "s": ["a", None], "f": [1.5, 2.0], "l": [[1], []], "st": [{"x": 1}, {"x": 2}],
        "day": [dt.date(2024, 1, 1), None], "at": [dt.datetime(2024, 1, 1, 3, 4, 5, 6), None],
        "tm": [dt.time(3, 4, 5), None], "took": [dt.timedelta(hours=3), None],
    }).with_columns(utc=pl.col("at").dt.replace_time_zone("UTC"))
    reloaded = FrameDoc.model_validate_json(FrameDoc.from_polars(frame).model_dump_json()).df
    assert reloaded.equals(frame) and reloaded.schema == frame.schema


def test_dates_nested_in_lists_arrays_and_structs_round_trip_through_json():
    day = dt.date(2024, 1, 1)
    frame = pl.DataFrame({
        "days": [[day, None], None], "pair": pl.Series([[day, day], None], dtype=pl.Array(pl.Date, 2)),
        "event": [{"on": day, "at": [dt.datetime(2024, 1, 1, 3)]}, None],
    })
    reloaded = FrameDoc.model_validate_json(FrameDoc.from_polars(frame).model_dump_json()).df
    assert reloaded.equals(frame) and reloaded.schema == frame.schema


def test_an_enum_keeps_its_category_order():
    doc = _through_json(PolarsSchema.from_polars(pl.Schema({"band": pl.Enum(["high", "low", "mid"])})).model_dump())
    assert doc == [["band", {"type": "Enum", "categories": ["high", "low", "mid"]}]]
    assert PolarsSchema.model_validate(doc).schema["band"].categories.to_list() == ["high", "low", "mid"]


def test_a_primitive_schema_reads_a_name_or_an_explicit_type():
    assert PrimitiveSchema.model_validate("Float64").polars_type == pl.Float64()
    explicit = {"type": "Array", "inner": "Int32", "shape": [4, 2]}
    assert PrimitiveSchema.model_validate(explicit).polars_type == pl.Array(pl.Int32, (4, 2))


def test_an_unknown_type_name_is_invalid():
    with pytest.raises(ValidationError, match="polars type"):
        PrimitiveSchema.model_validate("Floaty")
