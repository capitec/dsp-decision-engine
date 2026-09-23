import typing as t
import polars as pl
import polars.datatypes.classes as polars_dtypes
from pydantic import BaseModel, model_validator, PrivateAttr, Field
from .schema import TStruct, handle_type, convert_schema
from .dataframe import TDataFrameData, TDataFrameRow


class StructDefinition(BaseModel):
    # name: str
    fields: TStruct
    data: TDataFrameData
    display_field: str


class StructTypeDef(BaseModel):
    name: str
    type: t.Literal["struct"] = "struct"
    definition: StructDefinition
    # data: TDataFrameData

    _schema: pl.Schema = PrivateAttr()
    _dtype: polars_dtypes.DataType = PrivateAttr()
    _pl_df: pl.DataFrame = PrivateAttr()
    _indexed_values: t.Dict[str, TDataFrameRow] = PrivateAttr()

    @model_validator(mode="after")
    def construct_polars_df(self):
        # Convert definition to polars struct type (without type_defs to avoid circular lookups)
        struct_type = handle_type(self.definition.fields, type_defs=None)
        assert isinstance(
            struct_type, polars_dtypes.Struct
        ), "Expected StructTypeDef.definition to produce a Struct type"

        self._dtype = struct_type

        # Extract schema from struct fields
        fields = [
            (
                f.name if hasattr(f, "name") else f[0],
                f.dtype if hasattr(f, "dtype") else f[1],
            )
            for f in struct_type.fields
        ]
        self._schema = pl.Schema(fields)

        # Build dataframe with the schema
        try:
            self._pl_df = pl.DataFrame(self.definition.data, schema=self._schema)
        except Exception as e:
            raise ValueError(
                f"Could not load data into struct schema ({self.definition.fields}): {e}"
            ) from e

        indexed_values = {}
        for record in self.definition.data:
            key = record.get(self.definition.display_field)
            if key is None:
                raise ValueError(
                    f"Display field '{self.definition.display_field}' not found in record: {record}"
                )
            if key in indexed_values:
                raise ValueError(
                    f"Duplicate key '{key}' found for display field '{self.definition.display_field}' in struct data"
                )
            indexed_values[key] = record
        self._indexed_values = indexed_values
        return self

    def get_value_for_key(self, key: str) -> TDataFrameRow:
        value = self._indexed_values.get(key)
        if value is None:
            raise ValueError(
                f"Key '{key}' not found in struct definition '{self.name}'"
            )
        return value

    @property
    def df(self):
        return self._pl_df

    @property
    def schema(self):
        return self._schema

    @property
    def dtype(self):
        return self._dtype


class _CatDefinition(BaseModel):
    categories: t.List[str]


class CategoricalTypeDef(BaseModel):
    name: str
    type: t.Literal["categorical"] = "categorical"
    definition: _CatDefinition
    _dtype: polars_dtypes.DataType = PrivateAttr()

    @model_validator(mode="after")
    def construct_schema(self):
        # Create an Enum polars type with the pre-defined categories
        # Categories are already a list, so order is preserved
        self._dtype = polars_dtypes.Enum(self.definition.categories)
        return self

    @property
    def dtype(self):
        return self._dtype


TTypeDef = t.Annotated[
    t.Union[StructTypeDef, CategoricalTypeDef], Field(discriminator="type")
]


class ContainsDtypes(BaseModel):
    dtypes: TStruct
    type_defs: t.Dict[str, TTypeDef]

    _polars_schema: pl.Schema = PrivateAttr()

    @model_validator(mode="after")
    def _convert_schema(self) -> "t.Self":
        # Pass type_defs to handle_type so it can resolve type references
        schema = handle_type(self.dtypes, self.type_defs)
        assert isinstance(
            schema, polars_dtypes.Struct
        ), "Expected upper level to be a struct."
        try:
            fields = [
                (
                    f.name if hasattr(f, "name") else f[0],
                    f.dtype if hasattr(f, "dtype") else f[1],
                )
                for f in schema.fields
            ]
            self._polars_schema = pl.Schema(fields)
        except pl.exceptions.DuplicateError as e:
            raise ValueError(
                f"Found one or more duplicate keys in a struct field. Detail: {e}"
            )
        return self

    @property
    def schema(self):
        return self._polars_schema
