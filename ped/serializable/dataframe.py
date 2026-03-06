import typing as t
import polars as pl
from pydantic import BaseModel, model_validator, model_serializer, PrivateAttr
from .schema import PolarsSchema

TDataFrameRow = t.Dict[str,t.Any]
TDataFrameData = t.List[TDataFrameRow]

def build_polars_df(
    data: TDataFrameData,
    schema: t.Optional[PolarsSchema] = None,
) -> pl.DataFrame:
    """Construct a Polars DataFrame from raw dict data and an optional PolarsSchema,
    raising a descriptive ValueError on any construction failure."""
    try:
        if schema:
            return pl.DataFrame(data, schema=schema.schema)
        return pl.DataFrame(data)
    except Exception as e:
        schema_str = str(schema.root) if schema else "inferred"
        raise ValueError(
            f"Could not load data into schema ({schema_str}): {e}"
        ) from e


class DataFrame(BaseModel):
    data: TDataFrameData
    schema: t.Optional[PolarsSchema] = None

    _pl_df: pl.DataFrame = PrivateAttr()

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame):
        data = df.to_dicts()
        ret = cls(data=data)
        ret.infer_schema()
        return ret

    @model_validator(mode='before')
    @staticmethod
    def enable_raw_data(data:t.Any):
        if isinstance(data, list): return {"data": data}
        return data
    
    @model_validator(mode="after")
    def construct_polars_df(self):
        self._pl_df = build_polars_df(self.data, self.schema)
        return self

    @model_serializer
    def serialize(self) -> t.Any:
        if self.schema is None:
            return self.data
        return {"data": self.data, "schema": self.schema.model_dump()}

    def infer_schema(self):
        self.schema = PolarsSchema.from_polars_schema(self.df.collect_schema())

    @property
    def df(self):
        return self._pl_df
