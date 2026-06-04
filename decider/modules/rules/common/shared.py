import typing as t
import polars as pl
from pydantic import BaseModel, PrivateAttr, model_validator, Field
from ..serializable.dataframe import TDataFrameData, TDataFrameRow
from ..serializable.dtypes import ContainsDtypes, TTypeDef
from ..serializable.schema import TType, ExplicitType


class InputRef(BaseModel):
    """Reference to a runtime parameter or variable.

    Used to dynamically get parameters from the payload or dataframe at execution time.
    Represents variables in the UI (displayed as #key).
    """

    key: str = Field(description="Parameter key from the graph execution context")

    def resolve(self, parameters: t.Optional[pl.Expr]):
        """Resolve the reference to a Polars expression at runtime."""
        return parameters.struct.field(self.key)

    def __str__(self) -> str:
        """Display as #key for UI rendering."""
        return f"#{self.key}"


def expand_struct_type(
    struct_data: TDataFrameRow,
    dtype_keys: t.Iterable[t.Tuple[str, TTypeDef]],
    type_defs: t.Dict[str, TTypeDef],
) -> TDataFrameRow:
    out = {
        k: expand_data_row(struct_data[k], v, type_defs)
        for k, v in dtype_keys
        if k in struct_data
    }
    if set(out.keys()) != set(struct_data.keys()):
        raise ValueError(
            f"Miss-match in schema between struct data and dtype definition:\nStruct data keys: {set(struct_data.keys())}\nDtype keys: {set(out.keys())}"
        )
    return out


def expand_str(
    struct_data: t.Any, dtype: str, type_defs: t.Dict[str, TTypeDef]
) -> t.Any:
    return struct_data


def expand_explicit_type(
    struct_data: t.Any, dtype: ExplicitType, type_defs: t.Dict[str, TTypeDef]
) -> t.Any:
    lower_type = dtype.type.lower()
    if lower_type == "custom":
        type_id = dtype.model_extra.get("type_id")
        if not type_id:
            raise ValueError("ExplicitType of type 'Custom' must have a 'type_id'.")
        type_def = type_defs.get(type_id)
        if not type_def:
            raise ValueError(f"Type definition for {type_id} not found in type_defs.")

        if type_def.type == "categorical":
            return struct_data
        if type_def.type == "struct":
            # Handle case where struct_data is already an index (int/str) instead of a dict
            if isinstance(struct_data, (int, str)):
                # If it's already an index/key, use it directly - return the full record
                return type_def.get_value_for_key(struct_data)
            # If it's already the full expanded record dict, return it as-is
            # (Check if it matches the struct schema fields)
            if isinstance(struct_data, dict):
                # If it has all the fields from the struct schema, it's already expanded
                struct_fields = set(
                    type_def.definition.fields.keys()
                    if isinstance(type_def.definition.fields, dict)
                    else [f[0] for f in type_def.definition.fields]
                )
                if struct_fields == set(struct_data.keys()):
                    # Already expanded - return as-is
                    return struct_data
                # Otherwise, check for $key field
                key = struct_data.get("$key")
                if key is not None:
                    return type_def.get_value_for_key(key)
            raise ValueError(
                f"Struct data must be int, str, dict with '$key' field, or already-expanded struct dict, got: {struct_data}"
            )
        raise ValueError(
            f"Unsupported custom type {type_def.type} for ExplicitType with type_id {type_id}"
        )
    if lower_type in ("list", "set", "array"):
        if not isinstance(struct_data, t.Iterable):
            return struct_data
            # raise ValueError(f"Expected a list for {dtype.type} type, got {type(struct_data)}")
        inner_dtype = dtype.model_extra.get(
            "inner", dtype.model_extra.pop("fields", None)
        )
        return [expand_data_row(item, inner_dtype, type_defs) for item in struct_data]
    return expand_str(struct_data, dtype.type, type_defs)


def expand_data_row(
    tree_output: t.Any, dtype: TType, type_defs: t.Dict[str, TTypeDef]
) -> TDataFrameRow:
    if isinstance(dtype, dict):
        return expand_struct_type(tree_output, dtype.items(), type_defs)
    if isinstance(dtype, list):
        return expand_struct_type(tree_output, dtype, type_defs)
    if isinstance(dtype, str):
        return expand_str(tree_output, dtype, type_defs)
    if isinstance(dtype, ExplicitType):
        return expand_explicit_type(tree_output, dtype, type_defs)
    # This case should already be handled by the pydantic validator
    raise ValueError(
        f"Unexpected value {t}. Expected either a dict, list, string or explicit type"
    )


def expand_data(
    data: TDataFrameData, dtype: TType, type_defs: t.Dict[str, TTypeDef]
) -> t.List[TDataFrameRow]:
    return [expand_data_row(row, dtype, type_defs) for row in data]


class TreeOutput(ContainsDtypes):
    data: TDataFrameData
    default: t.Optional[TDataFrameRow] = None

    _output_literals: t.List[pl.Expr] = PrivateAttr()
    _default_literal: t.Optional[pl.Expr] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def construct_literals(self):
        # Build df purely to validate and resolve the struct dtype
        # Use parent's schema (from ContainsDtypes) which already handles type_defs

        data = expand_data(self.data, self.dtypes, self.type_defs)
        default = (
            expand_data_row(self.default, self.dtypes, self.type_defs)
            if self.default
            else None
        )
        try:
            pl_df = pl.DataFrame(data, schema=self.schema)
        except Exception as e:
            # Below was for when we didnt have schemas
            # if not len(data) and self.default is not None:
            #     dtypes = pl.DataFrame([self.default]).schema
            #     pl_df = pl.DataFrame(data, schema=dtypes)
            # else:
            raise ValueError(f"Could not load output data into schema: {e}") from e

        struct_dtype = pl.Struct(pl_df.schema)
        self._output_literals = [
            pl.lit(row, dtype=struct_dtype) for row in pl_df.to_dicts()
        ]
        if default is not None:
            # Validate default is compatible with schema
            try:
                pl.DataFrame([default], schema=self.schema)
            except Exception as e:
                raise ValueError(f"Default row is incompatible with schema: {e}") from e
            self._default_literal = pl.lit(default, dtype=struct_dtype)
        return self

    @property
    def output_literals(self) -> t.List[pl.Expr]:
        return self._output_literals

    @property
    def default_literal(self) -> t.Optional[pl.Expr]:
        return self._default_literal


class WithTreeOutput(BaseModel):
    output: TreeOutput

    @property
    def output_literals(self) -> t.List[pl.Expr]:
        return self.output.output_literals

    @property
    def default_literal(self) -> t.Optional[pl.Expr]:
        return self.output.default_literal
