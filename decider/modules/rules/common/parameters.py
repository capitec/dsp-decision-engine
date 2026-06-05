import typing as t
import polars as pl
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from ....serializable.schema import PrimitiveSchema


class ParameterInfo(BaseModel):
    type: PrimitiveSchema
    default_value: t.Optional[t.Any] = None
    _polars_literal: t.Optional[pl.Expr] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            polars_type = self.type.polars_type
            try:
                self._polars_literal = pl.lit(self.default_value, polars_type)
            except pl.exceptions.InvalidOperationError:
                raise ValueError(
                    f"Default value {self.default_value} is not compatible with type {self.type}"
                )

        return self

    @property
    def polars_literal(self) -> t.Optional[pl.Expr]:
        return self._polars_literal


class WithParameters:
    parameters_col: str = "parameters"
    parameters: t.Dict[str, ParameterInfo] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self):
        required_parameters = self.get_required_parameters()
        known_parameters = set(self.parameters.keys())
        if not required_parameters.issubset(known_parameters):
            missing = required_parameters - known_parameters
            raise ValueError(f"Missing parameter definitions for: {missing}")
        return self

    @property
    def parameter_schema(self) -> pl.Schema:
        return pl.Schema(
            {name: info.type.polars_type for name, info in self.parameters.items()}
        )
