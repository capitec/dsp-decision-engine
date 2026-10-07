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
        if not isinstance(schema, polars_dtypes.Struct):
            raise ValueError("Expected upper level to be a struct.")
        try:
            self._polars_schema = pl.Schema([(f.name, f.dtype) for f in schema.fields])
        except pl.exceptions.DuplicateError as e:
            raise ValueError(
                f"Found one or more duplicate keys in a struct field. Detail: {e}"
            )
        return self

    @classmethod
    def from_polars(cls, schema: pl.Schema) -> "PolarsSchema":
        return cls(convert_schema(schema))

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
        key = "inner" if "inner" in extra_dict else "fields"
        inner_definition = extra_dict.pop(key, None)
        if inner_definition is None:
            raise ValueError(
                f"For nested type {t} expected either an 'inner' or a 'fields' config."
            )
        args = (handle_type(inner_definition),)
    if isinstance(extra_dict.get("shape"), list):
        # JSON has no tuples, and polars takes an Array's shape only as an int or a tuple.
        extra_dict["shape"] = tuple(extra_dict["shape"])
    try:
        return pl_type(*args, **extra_dict)
    except TypeError as e:
        raise ValueError(
            f"Could not construct type {t} from args {t.model_extra}. Got error: {e}."
        )


class PrimitiveSchema(RootModel):
    """One polars type that isn't a struct, written as JSON: `"Float64"` or `{"type": "List", "inner": "String"}`.

    Example::

        PrimitiveSchema.model_validate({"type": "Array", "inner": "Int32", "shape": [4]}).polars_type
    """

    root: t.Union[str, ExplicitType]

    _polars_type: pl.DataType = PrivateAttr()

    @model_validator(mode="after")
    def _convert_type(self) -> "t.Self":
        self._polars_type = handle_type(self.root)
        return self

    @property
    def polars_type(self) -> pl.DataType:
        return self._polars_type


# The constructor arguments a dtype needs to come back the same; `inner` is handled for every nested type.
_ARGS: t.Dict[str, t.Tuple[str, ...]] = {
    "Datetime": ("time_unit", "time_zone"),
    "Duration": ("time_unit",),
    "Decimal": ("precision", "scale"),
    "Enum": ("categories",),
}


def convert_schema(schema: pl.Schema) -> list:
    """A polars schema as the JSON `PolarsSchema` reads: `[[column, type], ...]`, in column order.

    Example::

        convert_schema(pl.Schema({"tags": pl.List(pl.String)}))   # [("tags", {"type": "List", "inner": "String"})]
    """
    return [(name, convert_dtype(dtype)) for name, dtype in schema.items()]


def convert_dtype(dtype: t.Any) -> t.Any:
    """A polars dtype as JSON: its name, `{"type": name, **arguments}`, or a struct's `[[field, type], ...]`."""
    if isinstance(dtype, type):
        return dtype.__name__
    if isinstance(dtype, polars_dtypes.Struct):
        return [(f.name, convert_dtype(f.dtype)) for f in dtype.fields]
    name = type(dtype).__name__
    args = {}
    if isinstance(dtype, polars_dtypes.NestedType):
        args["inner"] = convert_dtype(dtype.inner)
    if isinstance(dtype, polars_dtypes.Array):
        # A multi-dimensional array's inner type is itself an array, so only the outer size is this level's.
        args["shape"] = dtype.size
    for arg in _ARGS.get(name, ()):
        value = getattr(dtype, arg)
        if value is not None:
            args[arg] = value.to_list() if isinstance(value, pl.Series) else value
    return {"type": name, **args} if args else name
