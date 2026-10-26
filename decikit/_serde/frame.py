import datetime as dt
import typing as t
import polars as pl
from pydantic import BaseModel, TypeAdapter, model_validator, model_serializer
from .schema import PolarsSchema

TDataFrameRow = t.Dict[str, t.Any]
TDataFrameData = t.List[TDataFrameRow]


def build_polars_df(
    data: TDataFrameData,
    schema: t.Optional[PolarsSchema] = None,
) -> pl.DataFrame:
    """Construct a Polars DataFrame from raw dict data and an optional PolarsSchema,
    raising a descriptive ValueError on any construction failure."""
    try:
        if schema:
            return pl.DataFrame(_parse_temporal(data, schema.schema), schema=schema.schema)
        return pl.DataFrame(data)
    except Exception as e:
        schema_str = str(schema.root) if schema else "inferred"
        raise ValueError(f"Could not load data into schema ({schema_str}): {e}") from e


_TEMPORAL = {pl.Date: dt.date, pl.Datetime: dt.datetime, pl.Time: dt.time, pl.Duration: dt.timedelta}


def _parse_temporal(data: TDataFrameData, schema: pl.Schema) -> TDataFrameData:
    # JSON holds dates and times as ISO strings, which polars won't read into a temporal column.
    parse = _parser(pl.Struct(schema))
    return data if parse is None else [parse(row) for row in data]


def _parser(dtype: pl.DataType) -> t.Optional[t.Callable[[t.Any], t.Any]]:
    if type(dtype) in _TEMPORAL:
        adapter = TypeAdapter(_TEMPORAL[type(dtype)])
        return lambda v: adapter.validate_python(v) if isinstance(v, str) else v
    if isinstance(dtype, (pl.List, pl.Array)):
        inner = _parser(dtype.inner)
        if inner is not None:
            return lambda v: v if v is None else [inner(x) for x in v]
    if isinstance(dtype, pl.Struct):
        fields = {f.name: p for f in dtype.fields if (p := _parser(f.dtype))}
        if fields:
            return lambda v: v if v is None else {**v, **{k: p(v[k]) for k, p in fields.items() if k in v}}
    return None


class FrameDoc(BaseModel):
    data: TDataFrameData
    dtypes: t.Optional[PolarsSchema] = None

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> "FrameDoc":
        return cls(data=df.to_dicts(), dtypes=PolarsSchema.from_polars(df.schema))

    @model_validator(mode="before")
    @staticmethod
    def enable_raw_data(data: t.Any):
        if isinstance(data, list):
            return {"data": data}
        return data

    @model_validator(mode="after")
    def check_dtypes(self):
        # Without dtypes the owner declares them; inferring here would reject rows its schema accepts.
        if self.dtypes is not None:
            build_polars_df(self.data, self.dtypes)
        return self

    @model_serializer
    def serialize(self) -> t.Any:
        if self.dtypes is None:
            return self.data
        return {"data": self.data, "dtypes": self.dtypes.model_dump()}

    @property
    def df(self) -> pl.DataFrame:
        return build_polars_df(self.data, self.dtypes)
