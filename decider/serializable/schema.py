import inspect
import typing as t
import typing_extensions as te
from functools import cache

from pydantic import (
    RootModel,
    BaseModel,
    ConfigDict,
    model_validator,
    PrivateAttr,
    Discriminator,
    Tag,
)

import polars as pl
import polars.datatypes.classes as polars_dtypes
from polars.datatypes import try_parse_into_dtype


class ExplicitType(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str

    # A dict with a "type" key is always an explicit type, never a struct with a "type" field:
    # telling them apart by whether "type" names a polars dtype would turn a typo into a struct.
    @model_validator(mode="after")
    def _validate_inner(self) -> "t.Self":
        extra_dict = {**self.model_extra}
        key = "inner" if "inner" in extra_dict else "fields"
        if key not in extra_dict:
            return self
        self.model_extra[key] = _RootTType.model_validate(extra_dict[key]).root

        return self


# Named aliases, so pydantic can build the recursive schema.
TOrderedStructType = te.TypeAliasType(
    "TOrderedStructType", "t.List[t.Tuple[str,TType]]"
)
TUnorderedStructType = te.TypeAliasType("TUnorderedStructType", "t.Dict[str,TType]")
TStruct = te.TypeAliasType(
    "TStruct",
    "t.Union[TOrderedStructType, TUnorderedStructType]",
)


def _type_def_discriminator(type_def: t.Any) -> str:
    if isinstance(type_def, str):
        return "str"
    if isinstance(type_def, list):
        return "struct"
    if isinstance(type_def, ExplicitType):
        return "explicit"
    if isinstance(type_def, dict):
        if "type" in type_def:
            return "explicit"
        return "struct"
    raise ValueError(
        f"Could not determine type definition for {type_def}. Expected either a string, list, dict or explicit type definition."
    )


TType = t.Annotated[
    t.Union[
        t.Annotated[str, Tag("str")],
        t.Annotated[ExplicitType, Tag("explicit")],
        t.Annotated["TStruct", Tag("struct")],
    ],
    Discriminator(_type_def_discriminator),
]


class _RootTType(RootModel):
    root: TType


class PolarsSchema(RootModel):
    """A polars schema written as JSON: `{column: type}` or `[[column, type], ...]`.

    A type is a polars dtype name (`"Float64"`), an explicit type with its
    arguments (`{"type": "List", "inner": "String"}`) or a nested struct.

    Example::

        PolarsSchema.model_validate({"income": "Float64", "tags": {"type": "List", "inner": "String"}}).schema
    """

    root: TStruct

    _polars_schema: pl.Schema = PrivateAttr()

    @model_validator(mode="after")
    def _convert_schema(self) -> "t.Self":
        schema = handle_type(self.root)
        assert isinstance(
            schema, polars_dtypes.Struct
        ), "Expected upper level to be a struct."
        try:
            self._polars_schema = pl.Schema([(f.name, f.dtype) for f in schema.fields])
        except pl.exceptions.DuplicateError as e:
            raise ValueError(
                f"Found one or more duplicate keys in a struct field. Detail: {e}"
            )
        return self

    @property
    def schema(self):
        return self._polars_schema


# User-facing type names that are aliases of a polars type.
CUSTOM_TYPE_MAPPINGS: t.Dict[str, str] = {
    "Set": "List",
}


@cache
def get_allowed_types():
    allowed_types = {
        k.lower(): v
        for k, v in inspect.getmembers(
            polars_dtypes,
            lambda tcls: inspect.isclass(tcls)
            and issubclass(tcls, polars_dtypes.DataType),
        )
    }
    allowed_types.update(
        {k.lower(): allowed_types[v.lower()] for k, v in CUSTOM_TYPE_MAPPINGS.items()}
    )
    return allowed_types


def handle_type(t: "TType | TStruct") -> pl.DataType:
    # A dict is an unordered struct, a list an ordered one.
    if isinstance(t, dict):
        return polars_dtypes.Struct(handle_kv_pair(t.items()))
    if isinstance(t, list):
        return polars_dtypes.Struct(handle_kv_pair(t))
    if isinstance(t, str):
        return handle_str(t)
    if isinstance(t, ExplicitType):
        return handle_explicit_type(t)
    raise ValueError(
        f"Unexpected value {t}. Expected either a dict, list, string or explicit type"
    )


def handle_kv_pair(it: "t.Iterable[t.Tuple[str, TType | TStruct]]"):
    return [pl.Field(k, handle_type(v)) for k, v in it]


def get_type_from_str(t: str):
    return get_allowed_types().get(t.lower())


def handle_str(t: str):
    pl_type = get_type_from_str(t)
    if pl_type is not None:
        try:
            return pl_type()
        except TypeError:
            # A type that needs arguments may still parse from its string form.
            pass
    out_type = try_parse_into_dtype(t)
    if out_type is None:
        raise ValueError(f"Could not convert {t} into a polars type")
    return out_type


def handle_explicit_type(t: ExplicitType):
    pl_type = get_type_from_str(t.type)
    if pl_type is None:
        raise ValueError(
            f"Could not convert {t} into a polars type. No polars type matching {t.type}."
        )
    args = tuple()
    extra_dict = {**t.model_extra}
    if issubclass(pl_type, polars_dtypes.NestedType):
        inner_definition = extra_dict.pop("inner", extra_dict.pop("fields", None))
        if inner_definition is None:
            raise ValueError(
                f"For nested type {t} expected either an 'inner' or a 'fields' config."
            )
        args = (handle_type(inner_definition),)
    try:
        return pl_type(*args, **extra_dict)
    except TypeError as e:
        raise ValueError(
            f"Could not construct type {t} from args {t.model_extra}. Got error: {e}."
        )
