import json
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

if t.TYPE_CHECKING:
    from .dtypes import TTypeDef


class ExplicitType(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str

    # Could do some validator here to distinguish between  ExplicitType and TStruct based on if the type exists in polars_dtypes
    # However this could lead to confusing messages if someone makes a typo.
    # I would rather have a user have to type out {value:[("type": "Int64"), ]} if they wanted to create a struct with "type" as an attribute
    @model_validator(mode="after")
    def _validate_inner(self) -> "t.Self":
        extra_dict = {**self.model_extra}  # Make a shallow copy
        key = "inner" if "inner" in extra_dict else "fields"
        if key not in extra_dict:
            return self
        self.model_extra[key] = _RootTType.model_validate(extra_dict[key]).root

        return self


# This is needed to avoid recursion error
# see: https://docs.pydantic.dev/2.11/concepts/types/#named-recursive-types
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
    Discriminator(
        _type_def_discriminator
    ),  # This is used to distinguish between explicit types and structs when the explicit type has extra fields. If the type field matches a known polars type we treat it as an explicit type otherwise we treat it as a struct
]

TPrimitiveType = t.Union[str, ExplicitType]


class PrimitiveSchema(RootModel):
    root: TPrimitiveType
    _polars_type: pl.Schema = PrivateAttr()

    @model_validator(mode="after")
    def _convert_schema(self) -> "t.Self":
        self._polars_type = handle_type(self.root)
        return self

    @property
    def polars_type(self):
        return self._polars_type


class _RootTType(RootModel):
    root: TType


class PolarsSchema(RootModel):
    root: TStruct

    _polars_schema: pl.Schema = PrivateAttr()

    @model_validator(mode="after")
    def _convert_schema(self) -> "t.Self":
        schema = handle_type(self.root)
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

    @classmethod
    def from_polars_schema(cls, schema: pl.Schema) -> "PolarsSchema":
        """Create a PolarsSchema from an existing Polars Schema object."""
        return cls(root=convert_schema(schema))

    @property
    def schema(self):
        return self._polars_schema


"""
The following are constants that are used throughout the type conversion
"""

CUSTOM_TYPE_MAPPINGS: t.Dict[str, str] = {
    # Add any custom mappings from user-friendly type names to Polars type names here
    # For example, if you want to allow "integer" as an alias for "Int64":
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


@cache
def get_type_properties():
    allowed_types = get_allowed_types()
    # Maps class name → tuple of constructor parameter names (excluding self / *args / **kwargs).
    # Auto-built by inspecting each DataType's __init__ signature so it stays in sync with the
    # installed polars version.
    type_properties: t.Dict[str, t.Tuple[str, ...]] = {}
    for _cls_name, _cls in allowed_types.items():
        try:
            _sig = inspect.signature(_cls.__init__)
            _params = [
                p
                for p in _sig.parameters.values()
                if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            if _params:
                type_properties[_cls.__name__] = tuple(p.name for p in _params)
        except (ValueError, TypeError):
            pass
    return type_properties


"""
The Following functions are used to convert from the json format into a polars schema
"""


def handle_type(
    t: TType | "TStruct", type_defs: "t.Optional[t.Dict[str, TTypeDef]]" = None
) -> pl.DataType:
    if isinstance(
        t, dict
    ):  # Unordered struct (Technically dicts should be ordered post 3.7)
        return polars_dtypes.Struct(handle_kv_pair(t.items(), type_defs))
    if isinstance(t, list):  # Ordered struct
        return polars_dtypes.Struct(handle_kv_pair(t, type_defs))
    if isinstance(t, str):
        return handle_str(t)
    if isinstance(t, ExplicitType):
        return handle_explicit_type(t, type_defs)
    # This case should already be handled by the pydantic validator
    raise ValueError(
        f"Unexpected value {t}. Expected either a dict, list, string or explicit type"
    )


def handle_kv_pair(
    it: t.Iterable[t.Tuple[str, TType | "TStruct"]],
    type_defs: "t.Optional[t.Dict[str, TTypeDef]]" = None,
):
    return [pl.Field(k, handle_type(v, type_defs)) for k, v in it]


def get_type_from_str(t: str):
    return get_allowed_types().get(t.lower())


def handle_str(t: str):
    pl_type = get_type_from_str(t)
    if pl_type is not None:
        try:
            return pl_type()
        except TypeError:
            # Could not construct type without args
            # try to still see if polars can automatically get types
            pass
    out_type = try_parse_into_dtype(t)
    if out_type is None:
        raise ValueError(f"Could not convert {t} into a polars type")
    return out_type


def handle_explicit_type(
    t: ExplicitType, type_defs: "t.Optional[t.Dict[str, TTypeDef]]" = None
):
    # Check if this is a type reference (has type_id in extra fields)
    if t.type.lower() == "custom":
        if t.model_extra is None or "type_id" not in t.model_extra:
            raise ValueError(
                "ExplicitType of type 'Custom' must have a 'type_id' in its extra fields."
            )
        type_id = t.model_extra["type_id"]
        if type_defs is None:
            raise ValueError(
                f"Type reference with type_id {type_id} requires type_defs to be provided"
            )

        if type_id not in type_defs:
            raise ValueError(f"Type ID {type_id} not found in type_defs")

        # Simply return the dtype from the type definition
        return type_defs[type_id].dtype

    pl_type = get_type_from_str(t.type)
    if pl_type is None:
        raise ValueError(
            f"Could not convert {t} into a polars type. No polars type matching {t.type}."
        )
    args = tuple()
    extra_dict = {**t.model_extra}  # Make a shallow copy
    if issubclass(pl_type, polars_dtypes.NestedType):
        inner_definition = extra_dict.pop("inner", extra_dict.pop("fields", None))
        if inner_definition is None:
            raise ValueError(
                f"For nested type {t} expected either an 'inner' or a 'fields' config."
            )
        inner_schema = handle_type(inner_definition, type_defs)
        args = (
            inner_schema,
        )  # Always the first arg for now we can maybe map inner to inner and fields to fields if needs be
    try:
        return pl_type(*args, **extra_dict)
    except TypeError as e:
        raise ValueError(
            f"Could not construct type {t} from args {t.model_extra}. Got error: {e}."
        )


"""
The following functions do the reverse and convert from a polars schema back into a json format
"""


def convert_kv_types(
    itr: t.Iterable[t.Tuple[str, polars_dtypes.DataType]],
) -> TOrderedStructType:
    return [(k, convert_dtype(v)) for k, v in itr]


def convert_schema(schema: pl.Schema) -> "TStruct":
    """Convert a Polars Schema to the ordered TStruct representation."""
    return [(name, convert_dtype(dtype)) for name, dtype in schema.items()]


def convert_dtype(dtype: polars_dtypes.DataType) -> "TType | TStruct":
    """Convert a Polars DataType instance back to its TType | TStruct JSON representation."""
    # Struct → ordered list of (field_name, converted_dtype) pairs
    if isinstance(dtype, polars_dtypes.DataTypeClass):  # Unconstricted types
        return dtype.__name__
    if isinstance(dtype, polars_dtypes.Struct):
        return convert_kv_types((field.name, field.dtype) for field in dtype.fields)

    cls = type(dtype)
    cls_name = cls.__name__
    props = get_type_properties().get(cls_name)

    # No constructor parameters → simple string e.g. "Int64"
    if not props:
        return cls_name

    # Collect non-None property values, recursing into nested DataTypes
    extra: t.Dict[str, t.Any] = {}
    for prop in props:
        val = getattr(dtype, prop, None)
        if val is None:
            continue
        if isinstance(val, (polars_dtypes.DataType, polars_dtypes.DataTypeClass)):
            # e.g. List.inner, Array.inner
            extra[prop] = convert_dtype(val)
        else:
            try:
                json.dumps(val)
            except TypeError:
                # Its a bit hacky but we want to make sure we can serialise the value
                # Some options like Categorical have non-dumpable values
                # We can try support more and more here but we need to be sure we dont break if those types
                # are used
                pass
            else:
                extra[prop] = val

    # If all params were None/default, still just return the type name
    if not extra:
        return cls_name

    return ExplicitType(type=cls_name, **extra)
