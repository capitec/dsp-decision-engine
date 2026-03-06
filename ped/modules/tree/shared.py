

import typing as t
import polars as pl
from pydantic import BaseModel, PrivateAttr, model_validator
from .nodes import NodeType
from ped.serializable.dataframe import TDataFrameData, TDataFrameRow, PolarsSchema, build_polars_df


class WithTreeOutput(BaseModel):
    output: TDataFrameData
    schema: t.Optional[PolarsSchema] = None
    default: t.Optional[TDataFrameRow] = None

    _output_literals: t.List[pl.Expr] = PrivateAttr()
    _default_literal: t.Optional[pl.Expr] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def construct_literals(self):
        # Build df purely to validate and resolve the struct dtype
        pl_df = build_polars_df(self.output, self.schema)
        struct_dtype = pl.Struct(pl_df.schema)
        self._output_literals = [
            pl.lit(row, dtype=struct_dtype) for row in pl_df.to_dicts()
        ]
        if self.default is not None:
            build_polars_df([self.default], self.schema)  # validate compatibility
            self._default_literal = pl.lit(self.default, dtype=struct_dtype)
        return self

    @property
    def output_literals(self) -> t.List[pl.Expr]:
        return self._output_literals

    @property
    def default_literal(self) -> t.Optional[pl.Expr]:
        return self._default_literal

